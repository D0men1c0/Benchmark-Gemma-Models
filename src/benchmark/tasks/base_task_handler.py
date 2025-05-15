from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from utils.logger import setup_logger
import torch

class TaskHandler(ABC):
    """Abstract base class for task handlers."""

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the task handler.
        :param model: The model to be used for the task.
        :param tokenizer: The tokenizer to be used for the task.
        :param device: The device to run the model on (e.g., "cpu", "cuda").
        :param advanced_args: Additional arguments for the task handler.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = setup_logger(self.__class__.__name__)
        
        self.advanced_args = advanced_args if advanced_args is not None else {}

        self.truncation = self.advanced_args.get("truncation", True)
        self.padding = self.advanced_args.get("padding", True) 
        
        self.generate_max_length = self.advanced_args.get("generate_max_length", 512)
        self.tokenizer_max_length = self.advanced_args.get("tokenizer_max_length", self.generate_max_length)

        self.skip_special_tokens = self.advanced_args.get("skip_special_tokens", True)
        self.max_new_tokens = self.advanced_args.get("max_new_tokens", 50)
        self.num_beams = self.advanced_args.get("num_beams", 1)
        self.do_sample = self.advanced_args.get("do_sample", False)
        self.use_cache = self.advanced_args.get("use_cache", True)
        self.clean_up_tokenization_spaces = self.advanced_args.get("clean_up_tokenization_spaces", True)

        self.temperature = self.advanced_args.get("temperature")
        self.top_p = self.advanced_args.get("top_p")
        self.top_k = self.advanced_args.get("top_k")

        # --- PADDING TOKEN LOGIC ---
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f"Tokenizer for {getattr(self.tokenizer, 'name_or_path', 'unknown_tokenizer')} has no pad_token. "
                    f"Setting pad_token to eos_token ('{self.tokenizer.eos_token}')."
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                generic_pad_token = "[PAD]"
                self.logger.warning(
                    f"Tokenizer for {getattr(self.tokenizer, 'name_or_path', 'unknown_tokenizer')} has no pad_token and no eos_token. "
                    f"Adding a new pad_token: '{generic_pad_token}'."
                )
                self.tokenizer.add_special_tokens({'pad_token': generic_pad_token})
                if hasattr(self.model, 'resize_token_embeddings') and callable(self.model.resize_token_embeddings):
                     try:
                        self.model.resize_token_embeddings(len(self.tokenizer))
                        self.logger.info(f"Resized model token embeddings to {len(self.tokenizer)} due to new pad_token.")
                     except Exception as e:
                        self.logger.error(f"Could not resize token embeddings after adding pad_token: {e}")

        if hasattr(self.model, 'config') and self.model.config is not None:
            if self.model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

        is_encoder_decoder_model = getattr(self.model.config, "is_encoder_decoder", False)
        
        if not is_encoder_decoder_model and self.tokenizer.padding_side == 'right':
            # This model is likely decoder-only (e.g., GPT2, Llama, Gemma)
            self.tokenizer.padding_side = 'left'

        self.logger.debug(
            f"{self.__class__.__name__} initialized with tokenizer_max_length={self.tokenizer_max_length}, "
            f"max_new_tokens={self.max_new_tokens}, pad_token_id={self.tokenizer.pad_token_id}, "
            f"padding_side='{self.tokenizer.padding_side}'"
        )


    def _generate_text(self, input_prompts: List[str]) -> List[str]:
        """
        Generates text from a list of input prompts.
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

        current_pad_token_id = self.tokenizer.pad_token_id
        if current_pad_token_id is None: # Should have been set in __init__
            if self.tokenizer.eos_token_id is not None:
                self.logger.debug("_generate_text: pad_token_id is None, falling back to eos_token_id for generation.")
                current_pad_token_id = self.tokenizer.eos_token_id
            else:
                default_hf_pad_token_id = 50256 
                self.logger.error(
                    f"_generate_text: tokenizer.pad_token_id and tokenizer.eos_token_id are None. "
                    f"Using a default pad_token_id: {default_hf_pad_token_id}. This may lead to unexpected behavior."
                )
                current_pad_token_id = default_hf_pad_token_id

        generation_params = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
            "pad_token_id": current_pad_token_id,
        }

        if self.temperature is not None:
            generation_params["temperature"] = self.temperature
        if self.top_p is not None:
            generation_params["top_p"] = self.top_p
        if self.top_k is not None:
            generation_params["top_k"] = self.top_k
        
        if 'attention_mask' in inputs:
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
            self.logger.debug(
                f"Model generated {generated_tokens_with_prompt.shape[1]} tokens, while input was {input_ids_length}. "
                "This might mean no new text was generated, or only EOS/pad tokens."
            )
            newly_generated_tokens = torch.empty(
                (generated_tokens_with_prompt.shape[0], 0), 
                dtype=torch.long, 
                device=generated_tokens_with_prompt.device
            )

        generated_texts = self.tokenizer.batch_decode(
            newly_generated_tokens,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )
        return generated_texts

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        pass