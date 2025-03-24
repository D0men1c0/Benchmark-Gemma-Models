import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from transformers import PreTrainedTokenizerFast
from .base_task_handler import TaskHandler

class ClassificationTaskHandler(TaskHandler):
    """Handler for classification tasks."""

    def process_example(self, example: Dict[str, Any]) -> Tuple[int, Optional[Any]]:
        """
        Process a single example and return the prediction and label.

        :param example: A dictionary representing a single dataset example.
        :return: Tuple containing the prediction and label (label may be None).
        """
        input_text = example.get("text", "")
        label = example.get("label", None)

        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=self.truncation, padding=self.padding
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()

        return predicted_class, label
    
    def process_batch(self, batch: Dict[str, Any]) -> Tuple[list, list]:
        """
        Process a batch of examples (text and labels) efficiently.

        :param batch: A dictionary containing a batch of examples.
        :return: Tuple containing the predicted classes and labels.
        """
        input_texts = batch.get("text", [])
        labels = batch.get("label", [])

        # Tokenize the entire batch at once
        inputs = self.tokenizer(
            input_texts, return_tensors="pt", truncation=self.truncation, padding=self.padding
        ).to(self.device)

        # Process in one go to speed up the computation
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1).cpu().numpy()  # Move to CPU for later processing

        return predicted_classes.tolist(), labels


class GenerationTaskHandler(TaskHandler):
    """Handler for text generation tasks."""

    def process_example(self, example: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
        input_text = example.get("text", "")
        label = example.get("label", None)

        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=self.truncation, padding=self.padding
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, max_length=self.generate_max_length)
            generated_text = self.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=self.skip_special_tokens
            )

        return generated_text, label

    def process_batch(self, batch: Dict[str, Any]) -> Tuple[list, list]:
        """
        Process a batch of examples for text generation efficiently.

        :param batch: A dictionary where each key corresponds to a field, and each value is a list.
        :return: A tuple containing a list of generated texts and a list of labels.
        """
        input_texts = batch.get("text", [])
        labels = batch.get("label", [])

        # Tokenize the batch
        inputs = self.tokenizer(
            input_texts, return_tensors="pt", truncation=self.truncation, padding=self.padding
        ).to(self.device)

        # Generate predictions for the batch
        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, max_length=self.generate_max_length)
            generated_texts = [self.tokenizer.decode(tokens, skip_special_tokens=self.skip_special_tokens) for tokens in generated_tokens]

        return generated_texts, labels