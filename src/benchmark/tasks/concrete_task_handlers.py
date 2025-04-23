import torch
from typing import Dict, Any, Tuple, List
from ..postprocessing.postprocessor_factory import PostProcessorFactory
from ..postprocessing.concrete_postprocessors import DefaultPostProcessor
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
    """Handles Multiple Choice QA tasks (e.g., MMLU)."""

    def _prepare_prompts(self, batch: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
        """Creates MMLU-style prompts and extracts raw labels."""
        questions = _ensure_list(batch.get('question', []))
        # Expects choices as a list of 4 lists (one per choice letter)
        choices_transposed = batch.get('choices', []) # Assume structure is correct or DatasetLoader fixed it
        answers = _ensure_list(batch.get('label_index', []))
        prompts = []
        raw_labels = []

        if not questions or not isinstance(choices_transposed, list) or len(choices_transposed) != 4:
            self.logger.warning("Invalid batch structure for MultipleChoice QA (questions or choices).")
            return [], []

        num_questions = len(questions)
        # Basic check for consistency
        if not all(len(c_list) == num_questions for c_list in choices_transposed) or len(answers) != num_questions:
             self.logger.warning(f"Length mismatch in MultipleChoice QA batch: Q={num_questions}, C={[len(c) for c in choices_transposed]}, A={len(answers)}")
             return [], []

        # Build prompts for valid items
        for i in range(num_questions):
            try:
                current_choices = [choices_transposed[j][i] for j in range(4)]
                choices_str = "\n".join([f"{chr(65+j)}) {choice}" for j, choice in enumerate(current_choices)])
                # Add a space after "Answer:" to potentially help model generation
                prompts.append(f"Question: {questions[i]}\nChoices:\n{choices_str}\nAnswer: ")
                raw_labels.append(answers[i])
            except IndexError:
                self.logger.warning(f"Index error processing item {i} in MultipleChoice QA batch. Skipping item.")
                continue # Skip this item if choices structure is wrong internally

        return prompts, raw_labels

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        input_prompts, raw_labels = self._prepare_prompts(batch)
        if not input_prompts:
            return [], [] # No valid prompts could be generated

        # Generate predictions using the base class method
        generated_outputs = self._generate_text(input_prompts)

        # Apply task-specific post-processing (e.g., extracting 'A', 'B', 'C', 'D')
        post_processor_key = "mmlu_generation" # Assumes MMLUPostProcessor uses this key
        try:
            processor = PostProcessorFactory.get_processor(post_processor_key)
            processed_predictions, processed_labels = processor.process(generated_outputs, raw_labels, batch)
        except Exception as e:
            self.logger.error(f"Post-processing failed for task '{post_processor_key}': {e}. Using DefaultPostProcessor.", exc_info=True)
            processor = DefaultPostProcessor()
            processed_predictions, processed_labels = processor.process(generated_outputs, raw_labels, batch)

        return processed_predictions, processed_labels


class MathReasoningGenerationTaskHandler(TaskHandler):
    """Handles Math Reasoning generation tasks (e.g., GSM8K)."""

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        input_texts = _ensure_list(batch.get('input_text', []))
        raw_labels = _ensure_list(batch.get('target_text', []))

        if not input_texts:
            return [], []

        # Generate using base method
        generated_texts = self._generate_text(input_texts)

        # Post-process (e.g., extract final answer for GSM8K)
        post_processor_key = "gsm8k" # Key for GSM8KPostProcessor
        try:
            processor = PostProcessorFactory.get_processor(post_processor_key)
            processed_predictions, processed_labels = processor.process(generated_texts, raw_labels, batch)
        except Exception as e:
            self.logger.error(f"Post-processing failed for task '{post_processor_key}': {e}. Using DefaultPostProcessor.", exc_info=True)
            processor = DefaultPostProcessor()
            processed_predictions, processed_labels = processor.process(generated_texts, raw_labels, batch)

        return processed_predictions, processed_labels


class SummarizationTaskHandler(TaskHandler):
    """Handles summarization tasks."""

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        input_texts = _ensure_list(batch.get('input_text', []))
        raw_labels = _ensure_list(batch.get('target_text', []))

        if not input_texts:
            return [], []

        # Generate using base method
        generated_texts = self._generate_text(input_texts)

        # Post-process (usually default or simple cleaning for summarization)
        post_processor_key = "summarization"
        try:
            processor = PostProcessorFactory.get_processor(post_processor_key)
            processed_predictions, processed_labels = processor.process(generated_texts, raw_labels, batch)
        except Exception as e:
            self.logger.error(f"Post-processing failed for task '{post_processor_key}': {e}. Using DefaultPostProcessor.", exc_info=True)
            processor = DefaultPostProcessor()
            processed_predictions, processed_labels = processor.process(generated_texts, raw_labels, batch)

        return processed_predictions, processed_labels


class TranslationTaskHandler(TaskHandler):
    """Handles translation tasks."""

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        input_texts = _ensure_list(batch.get('input_text', []))
        raw_labels = _ensure_list(batch.get('target_text', []))

        if not input_texts:
            return [], []

        # Optional: Add language prefixes if needed by the model
        # Example:
        # input_texts = [f"translate English to French: {text}" for text in input_texts]

        # Generate using base method
        generated_texts = self._generate_text(input_texts)

        # Post-process (usually default or simple cleaning)
        post_processor_key = "translation"
        try:
            processor = PostProcessorFactory.get_processor(post_processor_key)
            processed_predictions, processed_labels = processor.process(generated_texts, raw_labels, batch)
        except Exception as e:
            self.logger.error(f"Post-processing failed for task '{post_processor_key}': {e}. Using DefaultPostProcessor.", exc_info=True)
            processor = DefaultPostProcessor()
            processed_predictions, processed_labels = processor.process(generated_texts, raw_labels, batch)

        return processed_predictions, processed_labels


class ClassificationTaskHandler(TaskHandler):
    """Handles classification tasks."""

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[int], List[Any]]: # Predictions are int indices
        # This handler does NOT use text generation (_generate_text)
        input_texts = _ensure_list(batch.get('input_text', []))
        labels = _ensure_list(batch.get('label', [])) # Labels might be int or str depending on dataset

        if not input_texts or len(input_texts) != len(labels):
             self.logger.warning(f"Input texts are empty or length mismatch with labels in classification batch. Skipping.")
             return [], []

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.tokenizer_max_length # Use same max length for consistency
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Assume model output has 'logits' attribute
            try:
                logits = outputs.logits
                # Basic shape check
                if logits.dim() != 2 or logits.shape[0] != len(input_texts):
                     raise ValueError(f"Unexpected logits shape: {logits.shape}")
                predicted_classes = torch.argmax(logits, dim=-1).cpu().tolist()
            except (AttributeError, ValueError) as e:
                self.logger.error(f"Failed to get valid predictions from model output: {e}")
                return [], [] # Return empty predictions if logits are bad

        return predicted_classes, labels