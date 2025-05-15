from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from utils.logger import setup_logger
import torch

class TaskHandler(ABC):
    """Abstract base class for task handlers."""

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the task handler.

        :param model: Loaded model instance.
        :param tokenizer: Loaded tokenizer instance.
        :param device: Device to run the model on ('cpu' or 'cuda').
        :param advanced_args: Dictionary of advanced arguments from configuration,
                              including global settings and task-specific handler_options.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = setup_logger(self.__class__.__name__)
        
        # Store the entire advanced_args dictionary for subclasses to use
        self.advanced_args = advanced_args if advanced_args is not None else {}

        # Extract common tokenizer/generation parameters from advanced_args
        self.truncation = self.advanced_args.get("truncation", True)
        self.padding = self.advanced_args.get("padding", True) # Could be "longest", True, etc.
        
        # Max length for tokenizer when encoding inputs
        # Defaults to generate_max_length if not specified, which is a common use case.
        self.generate_max_length = self.advanced_args.get("generate_max_length", 512)
        self.tokenizer_max_length = self.advanced_args.get("tokenizer_max_length", self.generate_max_length)

        self.skip_special_tokens = self.advanced_args.get("skip_special_tokens", True) # For decoding
        self.max_new_tokens = self.advanced_args.get("max_new_tokens", 50) # For model.generate()
        self.num_beams = self.advanced_args.get("num_beams", 1)
        self.do_sample = self.advanced_args.get("do_sample", False)
        self.use_cache = self.advanced_args.get("use_cache", True) # Defaulting to True
        self.clean_up_tokenization_spaces = self.advanced_args.get("clean_up_tokenization_spaces", True)

        # Optional generation parameters, add more as needed
        self.temperature = self.advanced_args.get("temperature") # Will be None if not present
        self.top_p = self.advanced_args.get("top_p")
        self.top_k = self.advanced_args.get("top_k")

        self.logger.debug(f"{self.__class__.__name__} initialized with tokenizer_max_length={self.tokenizer_max_length}, max_new_tokens={self.max_new_tokens}")


    def _generate_text(self, input_prompts: List[str]) -> List[str]:
        """
        Generates text from a list of input prompts.
        Ensures only newly generated tokens (excluding prompt) are returned.
        :param input_prompts: List of input prompts to generate text from.
        :return: List of generated texts.
        """
        if not input_prompts:
            self.logger.warning("No input prompts provided to _generate_text.")
            return []
        
        inputs = self.tokenizer(
            input_prompts,
            return_tensors="pt",
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.tokenizer_max_length,
            add_special_tokens=True 
        ).to(self.device)

        generation_params = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
            "pad_token_id": self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None \
                            else self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
                            else 50256, # Fallback pad token ID
        }
        if self.temperature is not None:
            generation_params["temperature"] = self.temperature
        if self.top_p is not None:
            generation_params["top_p"] = self.top_p
        if self.top_k is not None:
            generation_params["top_k"] = self.top_k
        
        if 'attention_mask' in inputs: # Pass attention_mask if tokenizer provides it
            generation_params['attention_mask'] = inputs.attention_mask
        if 'token_type_ids' in inputs and inputs.token_type_ids is not None:
             generation_params['token_type_ids'] = inputs.token_type_ids

        self.logger.debug(f"Generating text with params: {generation_params} for {len(input_prompts)} prompts.")
        with torch.no_grad():
            generated_tokens_with_prompt = self.model.generate(
                input_ids=inputs.input_ids,
                **generation_params
            )
        
        input_ids_length = inputs.input_ids.shape[1]
        if generated_tokens_with_prompt.shape[1] > input_ids_length:
            newly_generated_tokens = generated_tokens_with_prompt[:, input_ids_length:]
        else:
            self.logger.debug(f"Model generated {generated_tokens_with_prompt.shape[1]} tokens, input was {input_ids_length}. Assuming no new text or only EOS.")
            # Create an empty tensor of shape (batch_size, 0) on the same device
            newly_generated_tokens = torch.empty((generated_tokens_with_prompt.shape[0], 0), dtype=torch.long, device=generated_tokens_with_prompt.device)

        generated_texts = self.tokenizer.batch_decode(
            newly_generated_tokens,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )
        return generated_texts

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        pass