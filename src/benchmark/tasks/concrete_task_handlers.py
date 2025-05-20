import importlib
from pathlib import Path
import torch
from typing import Dict, Any, Optional, Tuple, List
from ..postprocessing.postprocessor_factory import PostProcessorFactory
from ..postprocessing.concrete_postprocessors import DefaultPostProcessor
from ..prompting.prompt_builder_factory import PromptBuilderFactory
from .base_task_handler import TaskHandler

# Helper function to ensure data is list (simplifies handlers)
def _ensure_list(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, torch.Tensor):
        return data.tolist()
    # Handle single items or other iterables (like np.array implicitly converted)
    try:
        return list(data)
    except TypeError: # Handle non-iterable case (e.g., single int, str)
        return [data]


class MultipleChoiceQATaskHandler(TaskHandler):
    """
    Task handler for multiple-choice question answering tasks.
    This handler is designed to work with datasets that provide questions,
    choices, and labels in a structured format.
    It uses a prompt builder to create the appropriate prompts for the model.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the MultipleChoiceQATaskHandler.
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Dictionary of advanced arguments from configuration,
                              including global settings and task-specific handler_options.
        """
        super().__init__(model, tokenizer, device, advanced_args)
        prompt_template = self.advanced_args.get("prompt_template")
        builder_type = self.advanced_args.get("prompt_builder_type", "mmlu")
        builder_handler_args = {
            "default_subject": self.advanced_args.get("default_subject", "general knowledge"),
            "dataset_name": self.advanced_args.get("dataset_name"),
            "dataset_config_name": self.advanced_args.get("dataset_config_name"),
        }
        self.prompt_builder = PromptBuilderFactory.get_builder(
            builder_type=builder_type,
            prompt_template=prompt_template,
            handler_args=builder_handler_args
        )

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
        """
        Process a batch of multiple-choice questions.
        :param batch: Dictionary containing the batch data.
                      Expected keys: "question", "choices", "label_index", "subject".
        :return: Tuple of generated outputs and corresponding labels.
        """
        questions = _ensure_list(batch.get("question", []))
        choices_input = batch.get("choices", []) 
        labels = _ensure_list(batch.get("label_index", []))
        subjects = _ensure_list(batch.get("subject", [])) 

        num_items = len(questions)
        
        # Initial validation: must have questions and matching number of labels.
        if not (num_items > 0 and len(labels) == num_items):
            self.logger.warning(
                f"MCQA batch: Question count ({num_items}) and Label count ({len(labels)}) mismatch, "
                f"or no questions. Skipping batch."
            )
            return [], []

        # Validate and structure choices for each item
        item_structured_choices: List[List[str]] = [[] for _ in range(num_items)]

        if isinstance(choices_input, list) and choices_input:
            # Scenario 1: choices_input is item-wise [batch_size, num_options_per_item]
            # e.g., for MMLU, batch_size=8, num_options=4 -> len(choices_input) == 8
            if len(choices_input) == num_items and all(isinstance(ci, list) for ci in choices_input):
                self.logger.debug(f"MCQA choices format: item-wise (batch_size x num_options).")
                item_structured_choices = choices_input
            # Scenario 2: choices_input is transposed [num_options_total, batch_size]
            # e.g., for MMLU, num_options=4, batch_size=8 -> len(choices_input) == 4, and len(choices_input[0]) == 8
            elif len(choices_input) > 0 and isinstance(choices_input[0], list) and len(choices_input[0]) == num_items:
                choices_transposed = choices_input
                num_choice_options = len(choices_transposed) # Should be 4 for MMLU
                self.logger.debug(f"MCQA choices format: transposed (num_options x batch_size). Num_options: {num_choice_options}. Restructuring.")
                for item_idx in range(num_items):
                    current_item_s_choices = []
                    for choice_type_idx in range(num_choice_options):
                        try:
                            current_item_s_choices.append(choices_transposed[choice_type_idx][item_idx])
                        except IndexError:
                            self.logger.warning(f"IndexError restructuring choices for item {item_idx}, choice_type {choice_type_idx}.")
                            current_item_s_choices.append("CHOICE_READ_ERROR") # Placeholder
                    item_structured_choices[item_idx] = current_item_s_choices
            else:
                self.logger.warning(
                    f"MCQA choices structure is unrecognized or inconsistent for {num_items} questions. "
                    f"choices_input length: {len(choices_input)}. "
                    f"Type of choices_input[0]: {type(choices_input[0]).__name__ if choices_input else 'N/A'}. "
                    f"Content of choices_input[0] (first 50char): {str(choices_input[0])[:50] if choices_input and isinstance(choices_input[0], list) else 'N/A'}. "
                    f"Prompts may be missing choice information."
                )
                # item_structured_choices will remain list of empty lists.
        else:
            self.logger.warning("MCQA batch: 'choices' field is missing or not a list. Prompts will lack choices.")

        # Validate subjects length if subjects list is not empty
        if subjects and len(subjects) != num_items:
            self.logger.warning(f"MCQA batch: Subject count ({len(subjects)}) does not match question count ({num_items}). Using default subject where needed.")
            # Ensure subjects list is either empty or matches num_items for safe indexing later
            # This can be handled by how 'current_subject' is retrieved below.

        batch_items_for_prompting: List[Dict[str, Any]] = []
        valid_labels_for_processed_items: List[Any] = []

        for i in range(num_items):
            current_item_actual_choices = item_structured_choices[i]
            
            # If choices are essential for the prompt (e.g. default MMLU prompt) and they are missing/malformed for this item,
            # we might decide to skip this specific item.
            is_templated_prompt = bool(self.advanced_args.get("prompt_template"))
            if not current_item_actual_choices and not is_templated_prompt:
                 self.logger.warning(f"Skipping item {i} in MCQA due to missing/malformed choices and no custom prompt template (default MMLU prompt needs choices).")
                 continue # Skip this item from the batch

            formatted_choices_str = "\n".join(
                [f"{chr(65 + j)}) {choice_text}" for j, choice_text in enumerate(current_item_actual_choices)]
            )
            item_data = {
                "question": questions[i],
                "choices_formatted_str": formatted_choices_str,
                "subject": subjects[i] if i < len(subjects) and subjects else self.advanced_args.get("default_subject", "general knowledge"),
            }
            for j, choice_text in enumerate(current_item_actual_choices):
                if j < 26: # Max choices A-Z
                    item_data[f"choice_{chr(65+j)}"] = choice_text
            
            batch_items_for_prompting.append(item_data)
            valid_labels_for_processed_items.append(labels[i]) # Keep corresponding label
        
        if not batch_items_for_prompting:
            self.logger.warning("No items prepared for prompting in MCQA batch after all checks.")
            return [], [] 

        prompts = self.prompt_builder.build_prompts(batch_items_for_prompting)
        
        if not prompts:
            self.logger.warning(f"PromptBuilder returned no prompts for MCQA task.")
            # If prompts are empty, but we had items, return empty predictions and the valid labels
            return [], valid_labels_for_processed_items

        raw_outputs = self._generate_text(prompts)

        # Ensure the number of outputs matches the number of labels for processed items
        if len(raw_outputs) != len(valid_labels_for_processed_items):
            self.logger.error(f"Mismatch between number of generated outputs ({len(raw_outputs)}) and "
                               f"valid labels ({len(valid_labels_for_processed_items)}) for MCQA. "
                               "This indicates an issue in prompt generation or model generation for some items.")

        return self._post_process(raw_outputs, valid_labels_for_processed_items, batch)

    def _post_process(self, outputs: List[str], labels: List[Any], batch: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
        """
        Post-processes the outputs and labels.
        :param outputs: Raw generated text from the model for the batch.
        :param labels: Raw labels from the dataset batch.
        :param batch: The original batch data (optional).
        :return: Tuple of (processed_predictions, processed_labels).
                  Predictions are cleaned to extract the first valid letter (A, B, C, or D).
        """
        key = self.advanced_args.get("postprocessor_key", "mmlu_generation")
        try:
            proc = PostProcessorFactory.get_processor(key)
            return proc.process(outputs, labels, batch)
        except Exception as err:
            self.logger.error(f"Post-processing with key '{key}' failed: {err}. Falling back to DefaultPostProcessor.", exc_info=True)
            return DefaultPostProcessor().process(outputs, labels, batch)

class MathReasoningGenerationTaskHandler(TaskHandler):
    """
    Task handler for math reasoning generation tasks.
    This handler is designed to work with datasets that provide math problems
    and their corresponding solutions in a structured format.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the MathReasoningGenerationTaskHandler.
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Dictionary of advanced arguments from configuration,
                              including global settings and task-specific handler_options.
        """
        super().__init__(model, tokenizer, device, advanced_args)
        prompt_template = self.advanced_args.get("prompt_template")
        builder_type = self.advanced_args.get("prompt_builder_type", "default")
        builder_handler_args = self.advanced_args.copy() # Pass all advanced_args to builder
        self.prompt_builder = PromptBuilderFactory.get_builder(
            builder_type=builder_type,
            prompt_template=prompt_template,
            handler_args=builder_handler_args
        )

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Process a batch of math reasoning questions.
        :param batch: Dictionary containing the batch data.
                      Expected keys: "input_text", "target_text".
        :return: Dictionary with keys:
                 - "text_predictions": Generated reasoning and answers.
                    - "exact_match_predictions": Processed predictions for exact match.
        """
        inputs = _ensure_list(batch.get("input_text", []))
        labels_full_text = _ensure_list(batch.get("target_text", [])) 

        if not inputs:
            self.logger.warning("MathReasoning: No input_text found; returning empty outputs.")
            return {"text_predictions": [], "exact_match_predictions": [], "labels": []}
        
        batch_items_for_prompting = [{"input_text": q} for q in inputs]
        prompts = self.prompt_builder.build_prompts(batch_items_for_prompting)

        if not prompts:
            self.logger.warning("MathReasoning: No prompts generated.")
            return {"text_predictions": [], "exact_match_predictions": [], "labels": labels_full_text}

        # Generate text using the model
        raw_model_outputs = self._generate_text(prompts)

        # Ensure the number of outputs matches the number of labels
        exact_match_processed_predictions, exact_match_processed_labels = self._post_process_for_exact_match(raw_model_outputs, labels_full_text, batch)

        return {
            "text_predictions": raw_model_outputs, # Per ROUGE, BERTScore, DistinctNGram, WordEntropy
            "exact_match_predictions": exact_match_processed_predictions, # Per ExactMatchMetric
            "labels_for_text": labels_full_text, # Riferimenti per metriche testuali
            "labels_for_exact_match": exact_match_processed_labels # Riferimenti per exact match
        }

    def _post_process_for_exact_match(self, outputs: List[str], labels: List[str], batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Post-processes outputs specifically for GSM8K exact match on the final number.
        :param outputs: Raw generated text from the model for the batch.
        :param labels: Raw labels from the dataset batch.
        :param batch: The original batch data (optional).
        :return: Tuple of (processed_predictions, processed_labels).
                  Predictions are cleaned to extract the final number.
        """
        key = self.advanced_args.get("postprocessor_key", "gsm8k") # Should be "gsm8k"
        try:
            proc = PostProcessorFactory.get_processor(key)
            processed_predictions, processed_labels_for_em = proc.process(outputs, labels, batch)
            return processed_predictions, processed_labels_for_em
        except Exception as err:
            self.logger.error(f"Post-processing for exact_match with key '{key}' failed: {err}. Using raw outputs/labels.", exc_info=True)
            return outputs, [str(l) for l in labels]

class SummarizationTaskHandler(TaskHandler):
    """
    Task handler for summarization tasks.
    This handler is designed to work with datasets that provide articles
    and their corresponding summaries in a structured format.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the SummarizationTaskHandler.
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Dictionary of advanced arguments from configuration,
                              including global settings and task-specific handler_options.
        """
        super().__init__(model, tokenizer, device, advanced_args)
        prompt_template = self.advanced_args.get("prompt_template")
        if not prompt_template:
            prompt_template = "Summarize the following article:\n\n{article_text}\n\nSummary:" # Default
            self.logger.debug(f"SummarizationTaskHandler: Using default prompt template.")
        builder_type = self.advanced_args.get("prompt_builder_type", "default")
        builder_handler_args = self.advanced_args.copy()
        self.prompt_builder = PromptBuilderFactory.get_builder(
            builder_type=builder_type,
            prompt_template=prompt_template,
            handler_args=builder_handler_args
        )

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Process a batch of articles for summarization.
        :param batch: Dictionary containing the batch data.
        :return: Tuple of generated summaries and corresponding reference summaries.
        """
        articles = _ensure_list(batch.get("input_text", []))
        reference_summaries = _ensure_list(batch.get("target_text", []))
        if not articles:
            self.logger.warning("SummarizationTaskHandler: No articles found.")
            return [], []
        batch_items_for_prompting = [{"input_text": article, "article_text": article} for article in articles]
        prompts = self.prompt_builder.build_prompts(batch_items_for_prompting)
        if not prompts:
            self.logger.warning("SummarizationTaskHandler: No prompts generated.")
            return [], reference_summaries
        generated_summaries = self._generate_text(prompts)
        return self._post_process(generated_summaries, reference_summaries, batch)

    def _post_process(self, outputs: List[str], labels: List[str], batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Post-processes the outputs and labels.
        :param outputs: Raw generated text from the model for the batch.
        :param labels: Raw labels from the dataset batch.
        :param batch: The original batch data (optional).
        :return: Tuple of (processed_predictions, processed_labels).
                  Predictions are cleaned to extract the final summary.
        """
        key = self.advanced_args.get("postprocessor_key", "summarization")
        try:
            proc = PostProcessorFactory.get_processor(key)
            return proc.process(outputs, labels, batch)
        except Exception as e:
            self.logger.error(f"Post-processing with key '{key}' failed: {e}. Using DefaultPostProcessor.", exc_info=True)
            return DefaultPostProcessor().process(outputs, labels, batch)

class TranslationTaskHandler(TaskHandler):
    """
    Task handler for translation tasks.
    This handler is designed to work with datasets that provide source texts
    and their corresponding translations in a structured format.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the TranslationTaskHandler.
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Dictionary of advanced arguments from configuration,
                              including global settings and task-specific handler_options.
        """
        super().__init__(model, tokenizer, device, advanced_args)
        dataset_config_name = self.advanced_args.get("dataset_config_name", "src-tgt") # From BenchmarkRunner
        self.source_lang_code, self.target_lang_code = self._parse_language_codes(dataset_config_name)
        
        self.lang_code_to_name_map = {
            "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
        }

        builder_handler_args = self.advanced_args.copy()
        builder_handler_args.update({
            "source_lang_code": self.source_lang_code,
            "target_lang_code": self.target_lang_code,
            "source_lang_name": self._get_lang_name(self.source_lang_code),
            "target_lang_name": self._get_lang_name(self.target_lang_code),
        })
        
        prompt_template = self.advanced_args.get("prompt_template")
        builder_type = self.advanced_args.get("prompt_builder_type", "translation")
        self.prompt_builder = PromptBuilderFactory.get_builder(
            builder_type=builder_type,
            prompt_template=prompt_template,
            handler_args=builder_handler_args
        )

    def _parse_language_codes(self, config_name: str) -> Tuple[str, str]:
        """
        Parses the language codes from the dataset config name.
        :param config_name: The dataset config name (e.g., "en-fr").
        :return: Tuple of source and target language codes.
        """
        if config_name and "-" in config_name:
            parts = config_name.split("-", 1)
            if len(parts) == 2: return parts[0], parts[1]
        self.logger.warning(f"Invalid lang config: '{config_name}'. Defaulting to unknown.")
        return "unknown_source", "unknown_target"

    def _get_lang_name(self, lang_code: str) -> str:
        """
        Maps language codes to human-readable names.
        :param lang_code: The language code (e.g., "en", "fr").
        :return: Human-readable language name or the code itself if not found.
        """
        return self.lang_code_to_name_map.get(lang_code, lang_code.upper())

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Process a batch of translation tasks.
        :param batch: Dictionary containing the batch data.
                      Expected keys: "input_text", "target_text".
        :return: Tuple of generated translations and corresponding reference translations.
        """
        source_texts = _ensure_list(batch.get("input_text", []))
        references = _ensure_list(batch.get("target_text", []))
        if not source_texts:
            self.logger.warning("TranslationTaskHandler: No input_text.")
            return [], []
        batch_items_for_prompting = [{"input_text": text} for text in source_texts]
        prompts = self.prompt_builder.build_prompts(batch_items_for_prompting)
        if not prompts:
            self.logger.warning("TranslationTaskHandler: No prompts generated.")
            return [], references
        predictions = self._generate_text(prompts)
        return self._post_process(predictions, references, batch)

    def _post_process(self, outputs: List[str], labels: List[str], batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Post-processes the outputs and labels.
        :param outputs: Raw generated text from the model for the batch.
        :param labels: Raw labels from the dataset batch.
        :param batch: The original batch data (optional).
        :return: Tuple of (processed_predictions, processed_labels).
                  Predictions are cleaned to extract the final translation.
        """
        key = self.advanced_args.get("postprocessor_key", "translation")
        try:
            proc = PostProcessorFactory.get_processor(key)
            return proc.process(outputs, labels, batch)
        except Exception as e:
            self.logger.error(f"Post-processing with key '{key}' failed: {e}. Using DefaultPostProcessor.", exc_info=True)
            return DefaultPostProcessor().process(outputs, labels, batch)

class ClassificationTaskHandler(TaskHandler):
    """
    Task handler for classification tasks.
    This handler is designed to work with datasets that provide input texts
    and their corresponding labels in a structured format.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the ClassificationTaskHandler.
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Dictionary of advanced arguments from configuration,
                              including global settings and task-specific handler_options.
        """
        super().__init__(model, tokenizer, device, advanced_args)
        # self.tokenizer_max_length is set in BaseTaskHandler

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[int], List[Any]]:
        """
        Process a batch of classification tasks.
        :param batch: Dictionary containing the batch data.
                      Expected keys: "input_text", "label_index", "label".
        :return: Tuple of predicted classes and corresponding labels.
        """
        input_texts = _ensure_list(batch.get('input_text', []))
        labels = _ensure_list(batch.get('label_index', batch.get('label', [])))
        if not input_texts or len(input_texts) != len(labels):
            self.logger.warning(f"Classification batch mismatch/empty. Texts:{len(input_texts)}, Labels:{len(labels)}.")
            return [], []
        try:
            inputs = self.tokenizer(
                input_texts, return_tensors="pt", truncation=self.truncation,
                padding=self.padding, max_length=self.tokenizer_max_length
            ).to(self.device)
        except Exception as e:
            self.logger.error(f"Tokenization error in Classification: {e}", exc_info=True)
            return [], []
        with torch.no_grad():
            try:
                outputs = self.model(**inputs)
                logits = outputs.logits
                if logits.dim() != 2 or logits.shape[0] != len(input_texts):
                    raise ValueError(f"Logits shape mismatch: {logits.shape}")
                predicted_classes = torch.argmax(logits, dim=-1).cpu().tolist()
                return predicted_classes, labels
            except Exception as e:
                self.logger.error(f"Inference/logit processing error in Classification: {e}", exc_info=True)
                return [], []

class TextPairClassificationTaskHandler(TaskHandler):
    """
    Task handler for text pair classification tasks.
    This handler is designed to work with datasets that provide pairs of input texts
    and their corresponding labels in a structured format.
    """
    
    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the TextPairClassificationTaskHandler.
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Dictionary of advanced arguments from configuration,
                              including global settings and task-specific handler_options.
        """
        super().__init__(model, tokenizer, device, advanced_args)
        self.logger.info(f"TextPairClassificationTaskHandler initialized for: {self.advanced_args.get('dataset_config_name', 'Unknown Pair Task')}")

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[Any], List[Any]]: # Return type can be List[int] or List[float]
        """
        Process a batch of text pair classification tasks.
        :param batch: Dictionary containing the batch data.
                      Expected keys: "input_text_pair1", "input_text_pair2", "label_index", "label".
        :return: Tuple of predicted classes or scores and corresponding labels.
        """
        texts1 = _ensure_list(batch.get('input_text_pair1', []))
        texts2 = _ensure_list(batch.get('input_text_pair2', []))
        labels = _ensure_list(batch.get('label_index', batch.get('label', [])))
        if not texts1 or not texts2 or len(texts1) != len(texts2) or len(texts1) != len(labels):
            self.logger.warning(f"TextPairClassification batch mismatch/empty. T1:{len(texts1)}, T2:{len(texts2)}, L:{len(labels)}.")
            return [], []
        try:
            inputs = self.tokenizer(
                texts1, texts2, return_tensors="pt", truncation=self.truncation,
                padding=self.padding, max_length=self.tokenizer_max_length
            ).to(self.device)
        except Exception as e:
            self.logger.error(f"Tokenization error in TextPairClassification: {e}", exc_info=True)
            return [], []
        with torch.no_grad():
            try:
                outputs = self.model(**inputs)
                logits = outputs.logits
                if logits.dim() != 2 or logits.shape[0] != len(texts1):
                    raise ValueError(f"Logits shape mismatch: {logits.shape}")
                
                dataset_config = self.advanced_args.get("dataset_config_name", "")
                if dataset_config == "stsb": # GLUE STS-B (Regression)
                    if logits.shape[-1] != 1:
                        self.logger.error(f"STS-B expected 1 logit, got {logits.shape[-1]}.")
                        return [], []
                    predicted_scores = logits.squeeze(-1).cpu().tolist()
                    return predicted_scores, labels # Labels should be List[float]
                else: # Standard classification
                    predicted_classes = torch.argmax(logits, dim=-1).cpu().tolist()
                    return predicted_classes, labels
            except Exception as e:
                self.logger.error(f"Inference/logit processing error in TextPairClassification: {e}", exc_info=True)
                return [], []

class GlueClassificationPromptingTaskHandler(TaskHandler):
    """
    Task handler for single-sentence GLUE classification tasks using a prompting approach.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the GlueClassificationPromptingTaskHandler.
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Dictionary of advanced arguments from configuration,
                              including global settings and task-specific handler_options.
        """
        super().__init__(model, tokenizer, device, advanced_args)
        builder_type = self.advanced_args.get("prompt_builder_type")
        self.postprocessor_key = self.advanced_args.get("postprocessor_key")

        if not builder_type:
            raise ValueError(f"{self.__class__.__name__} requires 'prompt_builder_type' in handler_options.")
        if not self.postprocessor_key:
            raise ValueError(f"{self.__class__.__name__} requires 'postprocessor_key' in handler_options.")

        self.prompt_builder = PromptBuilderFactory.get_builder(
            builder_type=builder_type,
            prompt_template=self.advanced_args.get("prompt_template"),
            handler_args=self.advanced_args.copy()
        )
        self.logger.info(f"Initialized {self.__class__.__name__} with prompt_builder='{builder_type}', postprocessor='{self.postprocessor_key}'.")

    def _prepare_batch_items_for_prompting(self, batch: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """
        Helper to prepare items for prompting and extract original labels.
        :param batch: Dictionary containing the batch data.
        :return: Tuple of (batch_items_for_prompting, labels_original).
                 batch_items_for_prompting: List of dictionaries with input text.
                 labels_original: List of original labels.
        """
        input_texts = _ensure_list(batch.get("input_text", []))
        labels_original = _ensure_list(batch.get("label_index", batch.get("label", [])))

        if not input_texts or len(input_texts) != len(labels_original):
            self.logger.warning(
                f"{self.__class__.__name__} batch: Input/Label count mismatch or empty. "
                f"Texts:{len(input_texts)}, Labels:{len(labels_original)}. Skipping."
            )
            return [], []

        batch_items = []
        for text in input_texts:
            item_data = {"input_text": text}
            if "dataset_config_name" in self.advanced_args: # Pass context if available/needed
                 item_data["dataset_config_name"] = self.advanced_args["dataset_config_name"]
            batch_items.append(item_data)
        return batch_items, labels_original

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        """
        Process a batch of single-sentence classification data.
        :param batch: Dictionary containing the batch data.
                      Expected keys: "input_text", "label_index", "label".
        :return: Tuple of generated outputs and corresponding labels.
        """
        batch_items_for_prompting, labels_original = self._prepare_batch_items_for_prompting(batch)
        if not batch_items_for_prompting:
            return [], labels_original # Return original labels if no items to process, for accountability

        prompts = self.prompt_builder.build_prompts(batch_items_for_prompting)
        if not prompts:
            self.logger.warning(f"{self.__class__.__name__}: PromptBuilder returned no prompts.")
            return [], labels_original

        raw_model_outputs = self._generate_text(prompts)

        if len(raw_model_outputs) != len(labels_original):
            self.logger.error(
                f"{self.__class__.__name__}: Mismatch between #generated outputs ({len(raw_model_outputs)}) and "
                f"#labels ({len(labels_original)}). Trimming to shortest length."
            )
            min_len = min(len(raw_model_outputs), len(labels_original))
            raw_model_outputs = raw_model_outputs[:min_len]
            labels_to_process = labels_original[:min_len]
            if not raw_model_outputs: # If trimming resulted in no outputs
                return [], []
        else:
            labels_to_process = labels_original

        return self._post_process(raw_model_outputs, labels_to_process, batch)

    def _post_process(self, outputs: List[str], labels: List[Any], batch: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        """
        Post-processes the raw model outputs to extract class predictions.
        :param outputs: Raw generated text from the model for the batch.
        :param labels: Raw labels from the dataset batch.
        :param batch: The original batch data (optional).
        :return: Tuple of (processed_predictions, processed_labels).
                  Predictions are cleaned to extract the first valid letter (A, B, C, or D).
        """
        try:
            processor = PostProcessorFactory.get_processor(self.postprocessor_key)
            return processor.process(outputs, labels, batch)
        except Exception as err:
            self.logger.error(f"{self.__class__.__name__}: Post-processing with key '{self.postprocessor_key}' failed: {err}. Returning empty.", exc_info=True)
            return [], []


class GlueTextPairPromptingTaskHandler(GlueClassificationPromptingTaskHandler): # Can inherit for common logic
    """
    Task handler for GLUE text-pair classification/regression tasks using a prompting approach.
    """
    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        super().__init__(model, tokenizer, device, advanced_args) # Initializes prompt_builder, postprocessor_key etc.

    def _prepare_batch_items_for_prompting(self, batch: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """
        Helper to prepare items for prompting and extract original labels for text pairs.
        :param batch: Dictionary containing the batch data.
        :return: Tuple of (batch_items_for_prompting, labels_original).
                 batch_items_for_prompting: List of dictionaries with input text pairs.
                 labels_original: List of original labels.
        """
        texts1 = _ensure_list(batch.get("input_text_pair1", []))
        texts2 = _ensure_list(batch.get("input_text_pair2", []))
        labels_original = _ensure_list(batch.get("label_index", batch.get("label", [])))

        if not texts1 or not texts2 or len(texts1) != len(texts2) or len(texts1) != len(labels_original):
            self.logger.warning(
                f"{self.__class__.__name__} batch: Input/Label count mismatch or empty. "
                f"T1:{len(texts1)}, T2:{len(texts2)}, L:{len(labels_original)}. Skipping."
            )
            return [], []

        batch_items = []
        for i in range(len(texts1)):
            item_data = {
                "input_text_pair1": texts1[i],
                "input_text_pair2": texts2[i]
            }
            if "dataset_config_name" in self.advanced_args: # Pass context if available/needed
                 item_data["dataset_config_name"] = self.advanced_args["dataset_config_name"]
            batch_items.append(item_data)
        return batch_items, labels_original
    

class CreativeTextGenerationTaskHandler(TaskHandler):
    """
    Task handler for creative text generation tasks.
    Generates creative text based on an input prompt.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the CreativeTextGenerationTaskHandler.
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Dictionary of advanced arguments from configuration,
                              including global settings and task-specific handler_options.
        :param advanced_args: Dictionary of advanced arguments from configuration
        """
        super().__init__(model, tokenizer, device, advanced_args)
        prompt_template = self.advanced_args.get("prompt_template", "Write a creative piece about: {input_text}")
        builder_type = self.advanced_args.get("prompt_builder_type", "default")
        
        builder_handler_args = self.advanced_args.copy()
        if "dataset_name" in self.advanced_args:
            builder_handler_args["dataset_name"] = self.advanced_args["dataset_name"]

        self.prompt_builder = PromptBuilderFactory.get_builder(
            builder_type=builder_type,
            prompt_template=prompt_template,
            handler_args=builder_handler_args
        )
        self.logger.info(f"CreativeTextGenerationTaskHandler initialized with prompt_builder='{builder_type}'.")

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
        """
        Process a batch of creative text generation tasks.

        :param batch: Dictionary containing the batch data.
                      Expected key: "input_text" (il tema o l'incipit).
                      Potrebbe anche avere "target_text" se ci sono esempi di riferimento (opzionale).
        :return: Tuple of (list_of_generated_texts, list_of_reference_texts_or_inputs).
                 Se non ci sono target_text, reference_texts potrebbe essere una lista di None o degli input stessi.
        """
        input_prompts = _ensure_list(batch.get("input_text", []))
        reference_texts = _ensure_list(batch.get("target_text", [None] * len(input_prompts)))

        if not input_prompts:
            self.logger.warning("CreativeTextGenerationTaskHandler: No input_text found in batch.")
            return [], []

        batch_items_for_prompting = [{"input_text": text} for text in input_prompts]
        prompts_for_model = self.prompt_builder.build_prompts(batch_items_for_prompting)

        if not prompts_for_model:
            self.logger.warning("CreativeTextGenerationTaskHandler: PromptBuilder returned no prompts.")
            return [], reference_texts

        generated_texts = self._generate_text(prompts_for_model)

        return generated_texts, reference_texts
    
class CustomScriptTaskHandler(TaskHandler):
    """
    A task handler that delegates batch processing to a user-defined function
    in an external Python script.
    """
    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the CustomScriptTaskHandler.
        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Dictionary of advanced arguments from configuration,
                              including global settings and task-specific handler_options.
        """
        super().__init__(model, tokenizer, device, advanced_args)

        script_path_str = self.advanced_args.get("handler_script_path")
        self.function_name = self.advanced_args.get("handler_function_name")
        self.handler_script_args = self.advanced_args.get("handler_script_args", {})

        if not script_path_str or not self.function_name:
            raise ValueError("CustomScriptTaskHandler requires 'handler_script_path' and 'handler_function_name' in handler_options.")

        self.script_path = Path(script_path_str)
        if not self.script_path.is_file():
            raise FileNotFoundError(f"Handler script not found: {self.script_path}")

        self._load_custom_function()
        self.logger.info(f"CustomScriptTaskHandler initialized to use {self.function_name} from {self.script_path}")

    def _load_custom_function(self):
        """
        Load the custom function from the specified script.
        The function is expected to have the following signature:
        `my_function(batch, model, tokenizer, device, advanced_args, script_args) -> Tuple[List[Any], List[Any]]`
        """
        try:
            spec = importlib.util.spec_from_file_location(f"custom_handler_module.{self.function_name}", str(self.script_path))
            if spec is None or spec.loader is None: # Added check for spec.loader
                raise ImportError(f"Could not create module spec for handler script {self.script_path}")
            
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module) # type: ignore

            if not hasattr(custom_module, self.function_name):
                raise AttributeError(f"Function '{self.function_name}' not found in script '{self.script_path}'")
            
            self.custom_process_batch_fn = getattr(custom_module, self.function_name)
        except Exception as e:
            self.logger.error(f"Failed to load custom handler function '{self.function_name}' from '{self.script_path}': {e}", exc_info=True)
            raise


    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        """
        Processes a batch by calling the custom user-defined function to get prompts,
        then uses the framework-aware _generate_text method for model inference,
        and finally applies post-processing.

        The custom function (self.custom_process_batch_fn) is expected to have a signature like:
        `my_custom_logic(batch, model, tokenizer, device, advanced_args, script_args) -> Tuple[List[str], List[Any]]`
        It should return a tuple: (list_of_prompt_strings, list_of_raw_labels).
        The 'model', 'tokenizer', 'device' arguments are passed to the custom function for flexibility,
        though it might not use them if it's only generating prompts.

        :param batch: Dictionary containing the batch data.
        :return: Tuple of (processed_predictions, processed_labels).
        """
        if not hasattr(self, 'custom_process_batch_fn') or self.custom_process_batch_fn is None:
            self.logger.error(f"Custom batch processing function '{self.function_name}' not loaded for CustomScriptTaskHandler.")
            return [], []

        try:
            output_from_custom_fn = self.custom_process_batch_fn(
                batch=batch, 
                model=self.model, 
                tokenizer=self.tokenizer, 
                device=self.device,
                advanced_args=self.advanced_args, 
                script_args=self.handler_script_args
            )

            if not (isinstance(output_from_custom_fn, tuple) and len(output_from_custom_fn) == 2):
                self.logger.error(
                    f"Custom handler function '{self.function_name}' returned an unexpected type: {type(output_from_custom_fn)}. "
                    "Expected a tuple of (list_of_prompts, list_of_raw_labels)."
                )
                return [], []

            prompts_to_generate, raw_labels = output_from_custom_fn

            if not isinstance(prompts_to_generate, list):
                self.logger.error(
                    f"The first element (prompts) returned by custom function '{self.function_name}' is not a list "
                    f"(type: {type(prompts_to_generate)}). Cannot proceed with generation."
                )
                # Pass raw_labels through for consistent error handling if needed by metrics
                return [], raw_labels if isinstance(raw_labels, list) else []


            if not isinstance(raw_labels, list):
                self.logger.warning(
                    f"The second element (labels) returned by custom function '{self.function_name}' is not a list "
                    f"(type: {type(raw_labels)}). Using empty list for labels."
                )
                raw_labels = []


            raw_predictions: List[str] = []
            if not prompts_to_generate:
                self.logger.info(f"Custom handler function '{self.function_name}' returned an empty list of prompts. No text will be generated.")
                # Predictions will be an empty list, labels will be whatever the custom function returned.
            else:
                # Ensure all prompts are strings
                if not all(isinstance(p, str) for p in prompts_to_generate):
                    self.logger.error(
                        f"Not all items in the prompts list returned by custom function '{self.function_name}' are strings. "
                        "Cannot proceed with generation."
                    )
                    return [], raw_labels
                
                # Use the framework-aware _generate_text method from the base TaskHandler
                self.logger.debug(f"Calling self._generate_text with {len(prompts_to_generate)} prompts for framework {self.framework_type}.")
                raw_predictions = self._generate_text(prompts_to_generate)

            # Apply post-processing to the generated predictions and the labels from custom function
            return self._apply_post_processing(raw_predictions, raw_labels, batch)

        except Exception as e:
            self.logger.error(
                f"Error during execution of custom handler function '{self.function_name}' or "
                f"its subsequent generation/post-processing steps: {e}", exc_info=True
            )
            return [], []