from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from utils.logger import setup_logger
import torch
import tensorflow as tf
from ..postprocessing.postprocessor_factory import PostProcessorFactory
from ..postprocessing.concrete_postprocessors import DefaultPostProcessor

class TaskHandler(ABC):
    """
    Abstract base class for task handlers. Manages common initialization,
    text generation with framework-specific logic (PyTorch/TensorFlow)
    via helper methods, and post-processing.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str, advanced_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the task handler.

        :param model: The model instance for the task.
        :param tokenizer: The tokenizer instance for the task.
        :param device: Target device for PyTorch models (e.g., "cpu", "cuda").
        :param advanced_args: Dictionary of advanced arguments and task-specific options.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Determine framework type
        if hasattr(self.model, "hf_device_map") and "keras" in str(type(self.model)).lower():
            self.framework_type = "tensorflow"
        elif "tensorflow" in str(type(self.model)).lower() or "tf." in str(type(self.model)).lower() or "TF" in self.model.__class__.__name__:
            self.framework_type = "tensorflow"
        elif hasattr(self.model, "parameters") and callable(self.model.parameters):
            self.framework_type = "pytorch"
        else:
            self.framework_type = "pytorch"
            setup_logger(self.__class__.__name__).warning(
                f"Could not reliably determine model framework for {self.model.__class__.__name__}. Defaulting to PyTorch."
            )
        
        self.logger = setup_logger(f"{self.__class__.__name__}_{self.framework_type}")
        self.logger.info(f"TaskHandler initialized for a '{self.framework_type}' model.")
        
        self.advanced_args = advanced_args if advanced_args is not None else {}

        self.truncation = self.advanced_args.get("truncation", True)
        self.padding = self.advanced_args.get("padding", True)
        self.generate_max_length = self.advanced_args.get("generate_max_length", 512)
        self.tokenizer_max_length = self.advanced_args.get("tokenizer_max_length", self.generate_max_length)

        self.skip_special_tokens = self.advanced_args.get("skip_special_tokens", True)
        self.max_new_tokens = self.advanced_args.get("max_new_tokens", 50)
        self.clean_up_tokenization_spaces = self.advanced_args.get("clean_up_tokenization_spaces", True)

        self.num_beams = self.advanced_args.get("num_beams", 1)
        self.do_sample = self.advanced_args.get("do_sample", False)
        self.temperature = self.advanced_args.get("temperature")
        self.top_p = self.advanced_args.get("top_p")
        self.top_k = self.advanced_args.get("top_k")
        
        self.use_cache = self.advanced_args.get("use_cache", True)

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f"Tokenizer for '{getattr(self.tokenizer, 'name_or_path', 'unknown')}' has no pad_token. "
                    f"Setting pad_token to eos_token ('{self.tokenizer.eos_token}')."
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.logger.error(
                    f"CRITICAL: Tokenizer for '{getattr(self.tokenizer, 'name_or_path', 'unknown')}' "
                    "has no pad_token and no eos_token. Generation will likely fail."
                )
        
        if self.framework_type == "pytorch":
            if hasattr(self.model, 'config') and self.model.config is not None:
                if self.model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id
                is_encoder_decoder = getattr(self.model.config, "is_encoder_decoder", False)
                if not is_encoder_decoder and self.tokenizer.padding_side == 'right':
                    self.logger.debug("PyTorch Decoder-only model: Setting tokenizer padding_side to 'left'.")
                    self.tokenizer.padding_side = 'left'
            elif not hasattr(self.model, 'config') or self.model.config is None:
                 self.logger.warning("PyTorch model does not have 'config'. Skipping some PT-specific setups.")

        self.logger.debug(
            f"Initialized. Tokenizer max_length={self.tokenizer_max_length}, "
            f"max_new_tokens={self.max_new_tokens}, pad_token_id={self.tokenizer.pad_token_id}, "
            f"padding_side='{self.tokenizer.padding_side}'."
        )

    def _prepare_pytorch_inputs(self, input_prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Prepares tokenized inputs for PyTorch models.
        :param input_prompts: List of input prompts.
        :return: Dictionary of tokenized inputs as PyTorch tensors.
        """
        return self.tokenizer(
            input_prompts,
            return_tensors="pt",
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.tokenizer_max_length,
            add_special_tokens=True
        ).to(self.device)

    def _prepare_tensorflow_inputs(self, input_prompts: List[str]) -> Dict[str, tf.Tensor]:
        """
        Prepares tokenized inputs for TensorFlow models, converting from PT tensors.
        :param input_prompts: List of input prompts.
        :return: Dictionary of tokenized inputs as TensorFlow tensors.
        """
        pt_tokenized_on_device = self.tokenizer(
            input_prompts,
            return_tensors="pt", # Tokenize to PT first to leverage existing device placement
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.tokenizer_max_length,
            add_special_tokens=True
        ).to(self.device)

        tf_cpu_inputs: Dict[str, tf.Tensor] = {}
        for key, pt_gpu_tensor in pt_tokenized_on_device.items():
            numpy_array = pt_gpu_tensor.cpu().numpy()
            if key in ["input_ids", "attention_mask", "token_type_ids"]:
                tf_cpu_inputs[key] = tf.convert_to_tensor(numpy_array, dtype=tf.int32)
            else:
                tf_cpu_inputs[key] = tf.convert_to_tensor(numpy_array)
        return tf_cpu_inputs

    def _generate_with_pytorch(self, inputs_dict: Dict[str, torch.Tensor], generation_params: Dict[str, Any]) -> torch.Tensor:
        """
        Generates text using a PyTorch model.
        :param inputs_dict: Dictionary of tokenized inputs.
        :param generation_params: Dictionary of generation parameters.
        :return: Generated tokens as a PyTorch tensor.
        """
        self.logger.debug(f"Calling PT model.generate with params: {list(generation_params.keys())}")
        with torch.no_grad():
            return self.model.generate(
                input_ids=inputs_dict["input_ids"],
                **generation_params
            )

    def _generate_with_tensorflow(self, inputs_dict: Dict[str, tf.Tensor], generation_params: Dict[str, Any]) -> tf.Tensor:
        """
        Generates text using a TensorFlow model.
        :param inputs_dict: Dictionary of tokenized inputs.
        :param generation_params: Dictionary of generation parameters.
        :return: Generated tokens as a TensorFlow tensor.
        """
        self.logger.debug(f"Calling TF model.generate with params: {list(generation_params.keys())}")
        # TF generate expects input_ids and other inputs usually within the main call
        return self.model.generate(
            input_ids=inputs_dict["input_ids"],
            **generation_params
        )

    def _get_generation_parameters(self, inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares common and framework-specific generation parameters.
        :param inputs_dict: Dictionary of tokenized inputs.
        :return: Dictionary of generation parameters.
        """
        current_pad_token_id = self.tokenizer.pad_token_id
        if current_pad_token_id is None:
            current_pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            self.logger.warning(f"_get_generation_parameters: Using pad_token_id: {current_pad_token_id}.")

        gen_params: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "pad_token_id": int(current_pad_token_id),
        }
        if self.temperature is not None: gen_params["temperature"] = self.temperature
        if self.top_p is not None: gen_params["top_p"] = self.top_p
        if self.top_k is not None: gen_params["top_k"] = self.top_k
        
        if 'attention_mask' in inputs_dict and inputs_dict['attention_mask'] is not None:
            gen_params['attention_mask'] = inputs_dict['attention_mask']
        if 'token_type_ids' in inputs_dict and inputs_dict['token_type_ids'] is not None:
            gen_params['token_type_ids'] = inputs_dict['token_type_ids']

        if self.framework_type == "pytorch":
            gen_params["use_cache"] = self.use_cache
        elif self.framework_type == "tensorflow":
            gen_params.pop('use_cache', None) # TF might not use 'use_cache'
            # TF might prefer 'max_length' over 'max_new_tokens'. If issues arise:
            # input_ids_len = inputs_dict["input_ids"].shape[1]
            # gen_params["max_length"] = input_ids_len + self.max_new_tokens
            # gen_params.pop("max_new_tokens", None)
        return gen_params

    def _generate_text(self, input_prompts: List[str]) -> List[str]:
        """
        Generates text from input prompts, dispatching to framework-specific helpers.

        :param input_prompts: List of input prompts.
        :return: List of generated texts.
        """
        if not input_prompts:
            self.logger.warning("No input prompts provided to _generate_text.")
            return []

        tokenized_inputs: Dict[str, Any]
        generated_tokens_output: Any
        input_ids_len: int

        if self.framework_type == "tensorflow":
            tokenized_inputs = self._prepare_tensorflow_inputs(input_prompts)
            input_ids_tensor = tokenized_inputs["input_ids"]
            generation_call_params = self._get_generation_parameters(tokenized_inputs)
            generated_tokens_output = self._generate_with_tensorflow(tokenized_inputs, generation_call_params)
            input_ids_len = input_ids_tensor.shape[1]
            
            if generated_tokens_output.shape[1] > input_ids_len:
                newly_generated_tf_tensor = generated_tokens_output[:, input_ids_len:]
            else:
                self.logger.debug(f"TF: No new tokens or only EOS/pad.")
                newly_generated_tf_tensor = tf.constant([], shape=(generated_tokens_output.shape[0], 0), dtype=tf.int32)
            newly_generated_tokens_for_decode: Union[torch.Tensor, List[List[int]]] = [seq.numpy().tolist() for seq in newly_generated_tf_tensor]

        elif self.framework_type == "pytorch":
            tokenized_inputs = self._prepare_pytorch_inputs(input_prompts)
            input_ids_tensor = tokenized_inputs["input_ids"]
            generation_call_params = self._get_generation_parameters(tokenized_inputs)
            generated_tokens_output = self._generate_with_pytorch(tokenized_inputs, generation_call_params)
            input_ids_len = input_ids_tensor.shape[1]

            if generated_tokens_output.shape[1] > input_ids_len:
                newly_generated_tokens_for_decode = generated_tokens_output[:, input_ids_len:]
            else:
                self.logger.debug(f"PT: No new tokens or only EOS/pad.")
                newly_generated_tokens_for_decode = torch.empty(
                    (generated_tokens_output.shape[0], 0), dtype=torch.long, device=generated_tokens_output.device
                )
        else:
            self.logger.error(f"Unsupported framework '{self.framework_type}' in _generate_text.")
            return [f"ERROR: UNSUPPORTED FRAMEWORK '{self.framework_type}'"] * len(input_prompts)

        return self.tokenizer.batch_decode(
            newly_generated_tokens_for_decode,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )

    def _apply_post_processing(self, raw_predictions: List[Any], raw_labels: List[Any], original_batch: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], List[Any]]:
        """
        Applies post-processing using the configured post-processor.
        `self.advanced_args` (from handler_options) is used to get post-processor config.
        :param raw_predictions: Raw predictions from the model.
        :param raw_labels: Raw labels from the dataset.
        :param original_batch: Original batch of data (optional).
        :return: A tuple of (processed_predictions, processed_labels).
        """
        postprocessor_key = self.advanced_args.get("postprocessor_key")
        if not postprocessor_key:
            self.logger.debug(f"No 'postprocessor_key' in handler_options for {self.__class__.__name__}. Using DefaultPostProcessor.")
            return DefaultPostProcessor().process(raw_predictions, raw_labels, original_batch)

        # Prepare options for the PostProcessorFactory
        options_for_factory = {
            "script_path": self.advanced_args.get("postprocessor_script_path"),
            "function_name": self.advanced_args.get("postprocessor_function_name"),
            "script_args": self.advanced_args.get("postprocessor_script_args", {}),
        }
        # Merge general postprocessor options if they exist under 'postprocessor_options'
        if "postprocessor_options" in self.advanced_args:
            options_for_factory.update(self.advanced_args["postprocessor_options"])
        
        # Filter out None values for script_path and function_name as they are essential for CustomScriptPostProcessor init
        # but script_args can be an empty dict. Other options are passed as is.
        final_options_for_factory = {k: v for k, v in options_for_factory.items() 
                                     if not (k in ["script_path", "function_name"] and v is None)}


        self.logger.debug(f"Attempting to get post-processor with key '{postprocessor_key}' and options: {final_options_for_factory}")
        try:
            post_processor_instance = PostProcessorFactory.get_processor(
                postprocessor_key, 
                processor_options=final_options_for_factory
            )
            return post_processor_instance.process(raw_predictions, raw_labels, original_batch)
        except Exception as e:
            self.logger.error(f"Error during post-processing with key '{postprocessor_key}': {e}", exc_info=True)
            self.logger.warning("Falling back to DefaultPostProcessor due to error.")
            return DefaultPostProcessor().process(raw_predictions, raw_labels, original_batch)

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        """
        Processes a batch of data. Must be implemented by concrete handlers.
        Should call self._apply_post_processing at the end.
        :param batch: The input batch of data.
        :return: A tuple of (predictions, labels).
        """
        pass