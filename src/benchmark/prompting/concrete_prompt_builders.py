from typing import Dict, Any, List, Optional
from .base_prompt_builder import BasePromptBuilder
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TemplateBasedPromptBuilder(BasePromptBuilder):
    """
    Generic prompt builder: fills `self.template_string` or
    falls back to the item’s primary text field.
    """

    def build_prompts(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        """
        Builds a list of prompts from a list of batch items.
        Each item in batch_items is a dictionary representing one instance from the dataset.
        :param batch_items: A list of dictionaries, where each dictionary
        :return: A list of fully formatted prompt strings.
        """
        if not self.template_string:
            logger.warning("No template_string provided; using raw input_text/question.")
            return [
                str(item.get("input_text", item.get("question", "")))
                for item in batch_items
            ]

        return [
            self._format_single_prompt(
                self.template_string,
                item,
                fallback_text=str(item.get("input_text", item.get("question", "")))
            )
            for item in batch_items
        ]


class MMLUPromptBuilder(TemplateBasedPromptBuilder):
    """
    Builds MMLU‐style prompts. Expects `question` and either
    a pre‐formatted `choices_formatted_str` or raw `choices` list.
    """

    DEFAULT_TEMPLATE = "Question: {question}\nChoices:\n{choices_formatted_str}\nAnswer: "

    def build_prompts(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        """
        Builds MMLU‐style prompts. Expects `question` and either
        a pre‐formatted `choices_formatted_str` or raw `choices` list.
        :param batch_items: A list of dictionaries, where each dictionary
        :return: A list of fully formatted prompt strings.
        """
        tmpl = self.template_string or self.DEFAULT_TEMPLATE

        prompts = []
        for item in batch_items:
            data = item.copy()

            # If raw choices provided, format them
            if "choices" in data and "choices_formatted_str" not in data:
                choices = data.get("choices", [])
                formatted = "\n".join(f"{chr(65+j)}) {c}" for j, c in enumerate(choices))
                data["choices_formatted_str"] = formatted
                for j, c in enumerate(choices):
                    data[f"choice_{chr(65+j)}"] = c

            fallback = self.DEFAULT_TEMPLATE.format(
                question=data.get("question", "N/A"),
                choices_formatted_str=data.get("choices_formatted_str", "Choices N/A")
            )

            prompts.append(
                self._format_single_prompt(tmpl, data, fallback_text=fallback)
            )

        return prompts


class TranslationPromptBuilder(TemplateBasedPromptBuilder):
    """
    Builds translation prompts. Injects source/target codes & names into the template.
    """

    def build_prompts(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        """
        Builds translation prompts. Injects source/target codes & names into the template.
        :param batch_items: A list of dictionaries, where each dictionary
        :return: A list of fully formatted prompt strings.
        """
        src_code = self.handler_args.get("source_lang_code", "SRC")
        tgt_code = self.handler_args.get("target_lang_code", "TGT")
        src_name = self.handler_args.get("source_lang_name", src_code)
        tgt_name = self.handler_args.get("target_lang_name", tgt_code)

        default = f"Translate the following text from {src_name} to {tgt_name}: {{input_text}}"
        tmpl = self.template_string or default

        prompts = []
        for item in batch_items:
            data = {
                **item,
                "source_lang_code": src_code,
                "target_lang_code": tgt_code,
                "source_lang_name": src_name,
                "target_lang_name": tgt_name,
            }
            fallback = default.format(input_text=item.get("input_text", ""))
            prompts.append(
                self._format_single_prompt(tmpl, data, fallback_text=fallback)
            )

        return prompts
    

class GlueSST2PromptBuilder(BasePromptBuilder):
    """
    Prompt builder for GLUE SST-2 sentiment classification task.
    Prompts the LLM to classify sentiment.
    """
    DEFAULT_SST2_TEMPLATE = "Classify the sentiment of the following review as 'positive' or 'negative'. Do not add any other text. Review: \"{input_text}\"\nSentiment:"

    def build_prompts(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        """
        Builds a list of prompts for a batch of SST-2 items.

        :param batch_items: A list of dictionaries, where each dictionary
                            is expected to have an "input_text" key.
        :return: A list of fully formatted prompt strings.
        """
        prompts: List[str] = []
        # Use self.template_string if provided, otherwise use the default for this class
        template_to_use = self.template_string if self.template_string is not None else self.DEFAULT_SST2_TEMPLATE

        for item_data in batch_items:
            # Use the helper from BasePromptBuilder for formatting
            prompt = self._format_single_prompt(template_to_use, item_data, fallback_text="Error: Could not format SST-2 prompt.")
            prompts.append(prompt)
        return prompts


class GlueMRPCPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for GLUE MRPC paraphrase detection task.
    Prompts the LLM to determine if two sentences are paraphrases.
    """
    DEFAULT_MRPC_TEMPLATE = (
        "Are the following two sentences paraphrases? Answer ONLY with 'yes' or 'no'.\n"
        "Sentence 1: \"{input_text_pair1}\"\n"
        "Sentence 2: \"{input_text_pair2}\"\n"
        "Answer:"
    )

    def build_prompts(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        """
        Builds prompts for a batch of MRPC items.

        :param batch_items: A list of dictionaries, where each dictionary is expected
                            to have "input_text_pair1" and "input_text_pair2" keys.
        :return: A list of formatted prompts.
        """
        prompts: List[str] = []
        template_to_use = self.template_string if self.template_string is not None else self.DEFAULT_MRPC_TEMPLATE

        for item_data in batch_items:
            prompt = self._format_single_prompt(template_to_use, item_data, fallback_text="Error: Could not format MRPC prompt.")
            prompts.append(prompt)
        return prompts


class GlueSTSBPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for GLUE STS-B semantic textual similarity task.
    Prompts the LLM to predict a similarity score between 0.0 and 5.0.
    """
    DEFAULT_STSB_TEMPLATE = (
        "Rate the similarity of the following two sentences on a scale of 0.0 to 5.0. Provide ONLY the numerical score.\n"
        "Sentence 1: \"{input_text_pair1}\"\n"
        "Sentence 2: \"{input_text_pair2}\"\n"
        "Similarity Score:"
    )

    def build_prompts(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        """
        Builds prompts for a batch of STS-B items.

        :param batch_items: A list of dictionaries, where each dictionary is expected
                            to have "input_text_pair1" and "input_text_pair2" keys.
        :return: A list of formatted prompts.
        """
        prompts: List[str] = []
        template_to_use = self.template_string if self.template_string is not None else self.DEFAULT_STSB_TEMPLATE

        for item_data in batch_items:
            prompt = self._format_single_prompt(template_to_use, item_data, fallback_text="Error: Could not format STS-B prompt.")
            prompts.append(prompt)
        return prompts