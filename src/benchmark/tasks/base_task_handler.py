from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from utils.logger import setup_logger
import torch

class TaskHandler(ABC):
    """Abstract base class for task handlers assuming standardized input fields."""

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Any = None):
        """
        Initialize the task handler.

        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on.
        :param advanced_args: Additional arguments for the task handler.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = setup_logger(__name__)

        # Advanced arguments handling
        self.use_multi_gpu = advanced_args.get("use_multi_gpu", False) if advanced_args else False
        self.use_tpu = advanced_args.get("use_tpu", False) if advanced_args else False
        self.truncation = advanced_args.get("truncation", True) if advanced_args else True
        self.padding = advanced_args.get("padding", True) if advanced_args else True
        self.generate_max_length = advanced_args.get("generate_max_length", 512) if advanced_args else 512
        self.skip_special_tokens = advanced_args.get("skip_special_tokens", True) if advanced_args else True
        self.max_new_tokens = advanced_args.get("max_new_tokens", 50) if advanced_args else 50
        self.num_beams = advanced_args.get("num_beams", 1) if advanced_args else 1
        self.do_sample = advanced_args.get("do_sample", False) if advanced_args else False
        self.use_cache = advanced_args.get("use_cache", False) if advanced_args else False
        self.clean_up_tokenization_spaces = advanced_args.get("clean_up_tokenization_spaces", True) if advanced_args else True

        # Handling multi-GPU
        if self.use_multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        # Handling TPU
        if self.use_tpu:
            pass
            '''
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device()
            '''
            
    # Generate text using the model
    def _generate_text(self, input_texts: List[str]) -> List[str]:
        """
        Generate text using the model.
        :param input_texts: List of input texts to generate from.
        :return: List of generated texts.
        """
        if not input_texts:
            self.logger.warning("No input texts provided for generation.")
            return []
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.generate_max_length # Use relevant length param
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id or 50256, # Default pad token
            "use_cache": self.use_cache,
        }
        if 'attention_mask' in inputs:
            gen_kwargs['attention_mask'] = inputs['attention_mask']

        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=inputs['input_ids'],
                **gen_kwargs
            )

        generated_texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )
        return generated_texts

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Tuple[list, list]:
        """
        Process a batch of examples (texts and labels) efficiently.

        :param batch: A dictionary where each key corresponds to a field, and each value is a list.
        :return: A tuple of predictions and labels.
        """
        pass