from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import torch

class TaskHandler(ABC):
    """Abstract base class for task handlers."""

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

    @abstractmethod
    def process_example(self, example: Dict[str, Any]) -> Tuple[Any, Optional[Any]]:
        """
        Process a single example and return the prediction and label.

        :param example: A dictionary representing a single dataset example.
        :return: Tuple containing the prediction and label (label may be None).
        """
        pass
    
    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Tuple[list, list]:
        """
        Process a batch of examples (texts and labels) efficiently.

        :param batch: A dictionary where each key corresponds to a field, and each value is a list.
        :return: A tuple of predictions and labels.
        """
        raise NotImplementedError("This method should be implemented by subclasses if batch processing is needed.")