from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict


class ModelLoader:
    """
    Class for loading models from various checkpoints.

    :param checkpoint: Path or identifier for the model.
    :param framework: Framework to use for loading the model (e.g., "huggingface").
    """

    def __init__(self, checkpoint: str, framework: str):
        self.checkpoint = checkpoint
        self.framework = framework

    def load(self) -> Dict[str, Any]:
        """
        Loads the model and tokenizer from the checkpoint.

        :return: A dictionary with the model and tokenizer.
        """
        if self.framework == "huggingface":
            model = AutoModelForCausalLM.from_pretrained(self.checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            return {"model": model, "tokenizer": tokenizer}
        else:
            raise NotImplementedError(f"Framework {self.framework} not supported yet.")