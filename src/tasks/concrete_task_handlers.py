import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from .base_task_handler import TaskHandler

class ClassificationTaskHandler(TaskHandler):
    """Handler for classification tasks."""

    def process_example(self, example: Dict[str, Any]) -> Tuple[int, Optional[Any]]:
        input_text = example.get("text", "")
        label = example.get("label", None)

        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()

        return predicted_class, label


class GenerationTaskHandler(TaskHandler):
    """Handler for text generation tasks."""

    def process_example(self, example: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
        input_text = example.get("text", "")
        label = example.get("label", None)

        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, max_length=100)
            generated_text = self.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
            )

        return generated_text, label