import importlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TFAutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Tuple, Any, Optional
from .base_model_loader import BaseModelLoader
from utils.logger import setup_logger

class HuggingFaceModelLoader(BaseModelLoader):
    """
    Class for loading models from Hugging Face with optional quantization.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize the Hugging Face model loader.

        :param model_name: Name of the model to load.
        :param kwargs: Additional arguments for the model loader.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.model_name = model_name
        self.kwargs = kwargs

    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load a Hugging Face model and tokenizer with optional quantization.
        :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :return: A tuple containing the model and tokenizer.
        """
        self.logger.info(f"Loading model: {self.model_name} with quantization: {quantization} using HuggingFaceModelLoader")

        load_kwargs = self.kwargs.copy()
        enable_offloading = load_kwargs.pop('offloading', False)
        torch_dtype_str = load_kwargs.pop("torch_dtype", None)

        # Set default torch dtype
        actual_torch_dtype = torch.float32 # Correct: good default

        # Determine actual torch dtype (this logic is sound)
        if torch_dtype_str == "bfloat16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                actual_torch_dtype = torch.bfloat16
            else:
                self.logger.warning("Configured torch_dtype 'bfloat16' is not supported by CUDA, falling back to float32.")
        elif torch_dtype_str == "float16":
            actual_torch_dtype = torch.float16
        elif torch_dtype_str == "float32":
            actual_torch_dtype = torch.float32
        elif torch_dtype_str: # Specified but not recognized
            self.logger.warning(f"Unsupported torch_dtype '{torch_dtype_str}' specified. Defaulting to float32.")
        else: # torch_dtype_str is None (not specified in YAML)
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                actual_torch_dtype = torch.bfloat16 # Good to default to bfloat16 if available

        # Setup quantization config (this logic is sound for BNB)
        quantization_config = None # Renamed for clarity from your previous 'quantization_config_obj'
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=actual_torch_dtype,
                bnb_4bit_quant_type=load_kwargs.pop("bnb_4bit_quant_type", "nf4"), # Allows override
                bnb_4bit_use_double_quant=load_kwargs.pop("bnb_4bit_use_double_quant", True) # Allows override
            )
            load_kwargs["quantization_config"] = quantization_config
            self.logger.info(f"Using 4-bit quantization with compute dtype: {actual_torch_dtype}")

        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=actual_torch_dtype,
                bnb_8bit_use_double_quant=load_kwargs.pop("bnb_8bit_use_double_quant", True), # Allows override
                bnb_8bit_quant_type=load_kwargs.pop("bnb_8bit_quant_type", "dynamic") # Allows override, though "dynamic" is often implicit for 8-bit
            )
            load_kwargs["quantization_config"] = quantization_config
            self.logger.info(f"Using 8-bit quantization with compute dtype: {actual_torch_dtype}")

        # Handle offloading and device map
        if enable_offloading:
            self.logger.info("Offloading is enabled. Setting device_map='auto'.")
            load_kwargs["device_map"] = "auto"
        elif quantization_config is not None: # BNB quantization is active (4-bit or 8-bit)
            if "device_map" not in load_kwargs:
                load_kwargs["device_map"] = "auto"
                self.logger.info(f"BNB quantization active, device_map set to 'auto'.")
        else:
            if "device_map" in load_kwargs:
                self.logger.info(f"Using device_map from existing kwargs: '{load_kwargs['device_map']}'.")
            else:
                # If no device_map specified anywhere, it will load on default device.
                self.logger.info("No quantization, offloading, or explicit device_map. Model loads on default device(s).")

        load_kwargs.pop("quantization", None)

        final_from_pretrained_args = load_kwargs.copy()
        if quantization_config is None: 
            if torch_dtype_str or (actual_torch_dtype == torch.bfloat16 and actual_torch_dtype != torch.float32):
                final_from_pretrained_args["torch_dtype"] = actual_torch_dtype
                self.logger.info(f"Passing torch_dtype={actual_torch_dtype} to from_pretrained.")
        elif "torch_dtype" in final_from_pretrained_args:
            if final_from_pretrained_args["torch_dtype"] == str(actual_torch_dtype).split('.')[-1]: # Simple string compare
                final_from_pretrained_args.pop("torch_dtype")
                self.logger.info(f"torch_dtype in kwargs matched BNB compute_dtype, removed from from_pretrained args to avoid conflict.")


        self.logger.debug(f"Final args for AutoModelForCausalLM.from_pretrained: {sorted(list(final_from_pretrained_args.keys()))}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **final_from_pretrained_args
        )

        tokenizer_load_kwargs = {}
        if 'trust_remote_code' in self.kwargs: # Use self.kwargs for tokenizer, as load_kwargs might have popped it
            tokenizer_load_kwargs['trust_remote_code'] = self.kwargs['trust_remote_code']
        if 'revision' in self.kwargs:
            tokenizer_load_kwargs['revision'] = self.kwargs['revision']
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_load_kwargs)

        return model, tokenizer
    
class PyTorchModelLoader(BaseModelLoader):
    """
    Class for loading models using PyTorch with optional quantization.
    Enhanced to support bitsandbytes quantization and offloading.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize the PyTorch model loader.
        :param model_name: Name of the model to load.
        :param kwargs: Additional arguments for the model loader.
        """
        self.logger = setup_logger(f"{self.__class__.__name__}.{model_name}") # Log with model_name
        self.model_name = model_name
        self.kwargs = kwargs

    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load a PyTorch model and tokenizer with optional quantization.
        :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :return: A tuple containing the model and tokenizer.
        """
        self.logger.info(f"Loading model: {self.model_name} (Quant: {quantization}, Offload in config: {self.kwargs.get('offloading')}) using PyTorchModelLoader")
        load_kwargs = self.kwargs.copy()
        enable_offloading = load_kwargs.pop('offloading', False)
        torch_dtype_str = load_kwargs.pop("torch_dtype", None) # From ModelConfig's torch_dtype

        actual_torch_dtype = torch.float32 # Default
        if torch_dtype_str == "bfloat16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                actual_torch_dtype = torch.bfloat16
            else:
                self.logger.warning("Config: 'bfloat16' not CUDA supported for PyTorch, using float32.")
        elif torch_dtype_str == "float16":
            actual_torch_dtype = torch.float16
        elif torch_dtype_str == "float32":
            actual_torch_dtype = torch.float32
        elif torch_dtype_str:
            self.logger.warning(f"Config: Unsupported torch_dtype '{torch_dtype_str}' for PyTorch, using float32.")
        elif torch_dtype_str is None and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            actual_torch_dtype = torch.bfloat16
            self.logger.info(f"PyTorchLoader: torch_dtype not specified, defaulting to supported bfloat16.")

        quantization_config_for_bnb = None
        if quantization == "4bit":
            quantization_config_for_bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=actual_torch_dtype,
                bnb_4bit_quant_type=load_kwargs.pop("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=load_kwargs.pop("bnb_4bit_use_double_quant", True)
            )
            self.logger.info(f"PyTorchLoader: 4-bit quantization enabled. Compute dtype: {actual_torch_dtype}.")
        elif quantization == "8bit":
            quantization_config_for_bnb = BitsAndBytesConfig(load_in_8bit=True)
            self.logger.info("PyTorchLoader: 8-bit quantization enabled.")

        if quantization_config_for_bnb:
            load_kwargs["quantization_config"] = quantization_config_for_bnb

        device_map_from_kwargs = load_kwargs.pop("device_map", None)
        if quantization_config_for_bnb is not None:
            if device_map_from_kwargs is None:
                load_kwargs["device_map"] = "auto"
                self.logger.info(f"PyTorchLoader: BNB active, device_map set to 'auto'.")
            else:
                load_kwargs["device_map"] = device_map_from_kwargs
                self.logger.info(f"PyTorchLoader: BNB active, using pre-set device_map: '{device_map_from_kwargs}'.")
        elif enable_offloading:
            if device_map_from_kwargs is None:
                load_kwargs["device_map"] = "auto"
                self.logger.info(f"PyTorchLoader: Offloading:true, device_map set to 'auto'.")
            else:
                load_kwargs["device_map"] = device_map_from_kwargs
                self.logger.info(f"PyTorchLoader: Offloading:true, using pre-set device_map: '{device_map_from_kwargs}'.")
        elif device_map_from_kwargs is not None:
            load_kwargs["device_map"] = device_map_from_kwargs
            self.logger.info(f"PyTorchLoader: Using device_map from original kwargs: '{device_map_from_kwargs}'.")
        else:
            self.logger.info("PyTorchLoader: No BNB quantization, no explicit offloading, no device_map in kwargs. Model loads on default device(s).")

        if quantization_config_for_bnb is None:
            if torch_dtype_str or (actual_torch_dtype == torch.bfloat16 and torch.float32 != torch.bfloat16):
                load_kwargs["torch_dtype"] = actual_torch_dtype
                self.logger.info(f"PyTorchLoader: Not using BNB, passing torch_dtype={actual_torch_dtype} to from_pretrained.")
        elif "torch_dtype" in load_kwargs:
             self.logger.debug(f"PyTorchLoader: Using BitsAndBytes, but torch_dtype='{load_kwargs['torch_dtype']}' also found in load_kwargs. Will be passed to from_pretrained.")

        load_kwargs.pop("quantization", None)

        self.logger.debug(f"PyTorchLoader: Final args for AutoModelForCausalLM.from_pretrained: {sorted(list(load_kwargs.keys()))}")
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        tokenizer_load_args = {}
        for key in ["trust_remote_code", "revision", "use_fast"]:
            if key in self.kwargs: # Check original self.kwargs
                tokenizer_load_args[key] = self.kwargs[key]
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_load_args)

        return model, tokenizer

class TensorFlowModelLoader(BaseModelLoader):
    """
    Class for loading models using TensorFlow.
    Quantization with bitsandbytes is not applicable here.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize the TensorFlow model loader.
        :param model_name: Name of the model to load.
        :param kwargs: Additional arguments for the model loader.
        """
        self.logger = setup_logger(f"{self.__class__.__name__}.{model_name}") # Log with model_name
        self.model_name = model_name
        self.kwargs = kwargs

    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load a TensorFlow model and tokenizer.
        :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :return: A tuple containing the model and tokenizer.
        """
        self.logger.info(f"Loading model: {self.model_name} with quantization: {quantization} using TensorFlowModelLoader")
        
        load_kwargs = self.kwargs.copy()

        # Remove PyTorch-specific or BNB-specific args if they somehow ended up in kwargs
        load_kwargs.pop('torch_dtype', None)
        load_kwargs.pop('offloading', None) # 'offloading' as a boolean flag is tied to our PyTorch device_map logic
        load_kwargs.pop('device_map', None) # device_map in this form is for PyTorch/Accelerate

        # Handle quantization for TensorFlow (which is different from BNB)
        if quantization:
            self.logger.warning(
                f"TensorFlowModelLoader: Quantization '{quantization}' was specified. "
                "Note: BitsAndBytes quantization (4bit/8bit) is PyTorch-specific. "
                "TensorFlow uses different quantization methods (e.g., via TFLite, TFMOT). "
                "This loader will attempt to load the base TF model. "
                "Implement TensorFlow-specific quantization if needed."
            )
        # Remove 'quantization' string if it was in kwargs, as it's not directly used by TFAutoModel for BNB-style quant
        load_kwargs.pop("quantization", None)
        load_kwargs.pop("bnb_4bit_quant_type", None) # Clean up other potential BNB params
        load_kwargs.pop("bnb_4bit_use_double_quant", None)


        # TensorFlow models handle device placement differently (e.g. tf.distribute.Strategy)
        # from_pretrained for TF models typically loads to CPU by default or first available GPU
        # if TensorFlow is configured to use it. No direct "device_map" like PyTorch.
        self.logger.info("TensorFlowModelLoader: Device placement is handled by TensorFlow's default behavior or distribution strategies.")

        # Pass relevant args like trust_remote_code, revision
        tf_from_pretrained_args = {}
        for key in ["trust_remote_code", "revision", "from_pt"]: # "from_pt=True" if loading PyTorch weights into TF
            if key in load_kwargs:
                tf_from_pretrained_args[key] = load_kwargs[key]
        
        self.logger.debug(f"TensorFlowLoader: Final args for TFAutoModelForCausalLM.from_pretrained: {sorted(list(tf_from_pretrained_args.keys()))}")
        try:
            model = TFAutoModelForCausalLM.from_pretrained(self.model_name, **tf_from_pretrained_args)
        except Exception as e:
            self.logger.error(f"TensorFlowLoader: Failed to load model {self.model_name}. Error: {e}")
            # Check if it's a PyTorch model and if from_pt=True might help
            if "from_pt" not in tf_from_pretrained_args and "pytorch_model.bin" in str(e).lower(): # Basic check
                self.logger.info("TensorFlowLoader: Model might be PyTorch-native. Attempting with from_pt=True...")
                tf_from_pretrained_args["from_pt"] = True
                model = TFAutoModelForCausalLM.from_pretrained(self.model_name, **tf_from_pretrained_args)
            else:
                raise e # Re-throw original error if from_pt didn't help or wasn't applicable

        # Tokenizer loading is generally framework-agnostic
        tokenizer_load_args = {}
        for key in ["trust_remote_code", "revision", "use_fast"]:
            if key in self.kwargs: # Check original self.kwargs
                tokenizer_load_args[key] = self.kwargs[key]
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_load_args)

        return model, tokenizer
    

class CustomScriptModelLoader(BaseModelLoader):
    """
    Class for loading models using a user-defined function from a custom script.
    This is useful for loading models that are not directly supported by the
    Hugging Face Transformers library or for custom loading logic.
    """

    def __init__(self, model_name: str, framework: str, **kwargs: Any):
        """
        Initialize the CustomScriptModelLoader.
        This loader is designed to execute a user-defined function from a script
        to load a model and tokenizer.

        :param model_name: Name of the model to load.
        :param framework: Framework type (e.g., "custom_script").
        :param kwargs: Additional arguments for the model loader.
        """
        self.logger = setup_logger(f"{self.__class__.__name__}.{model_name}")
        self.descriptive_name = model_name
        self.framework_type = framework # Use the explicitly passed framework

        # Extract essential parameters from kwargs
        script_path_str = kwargs.pop('script_path', None)
        self.function_name = kwargs.pop('function_name', None)

        if self.framework_type != "custom_script":
            self.logger.warning(
                f"CustomScriptModelLoader for model '{model_name}' received framework '{self.framework_type}' "
                "instead of the expected 'custom_script'. Ensure factory logic is correct."
            )
        if not script_path_str:
            raise ValueError(f"'{self.__class__.__name__}' for model '{model_name}' requires 'script_path' to be in kwargs passed from factory.")
        if not self.function_name:
            raise ValueError(f"'{self.__class__.__name__}' for model '{model_name}' requires 'function_name' to be in kwargs passed from factory.")

        self.script_path = Path(script_path_str)
        if not self.script_path.is_file():
            raise FileNotFoundError(f"Custom model script not found at: {self.script_path}")

        # kwargs now contains all *other* args from YAML (like checkpoint, torch_dtype_str, script_args, etc.)
        # because script_path and function_name were popped.
        # 'framework' was an explicit param so it's not in kwargs here.
        self.constructor_passed_kwargs = kwargs

        self.logger.info(
            f"Initialized CustomScriptModelLoader for '{self.descriptive_name}': "
            f"script='{self.script_path}', function='{self.function_name}', framework='{self.framework_type}'"
        )
        self.logger.debug(f"  Constructor stored these other config keys from kwargs: {list(self.constructor_passed_kwargs.keys())}")
        if "script_args" in self.constructor_passed_kwargs:
            self.logger.debug(f"  Specific script_args from YAML: {self.constructor_passed_kwargs['script_args']}")

    def load(self, quantization: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load a model and tokenizer using a user-defined function from a custom script.
        :param quantization: Optional quantization type (e.g., "4bit", "8bit"). Defaults to None.
        :return: A tuple containing the model and tokenizer.
        """
        self.logger.info(f"Loading custom model '{self.descriptive_name}' using function '{self.function_name}' from script '{str(self.script_path)}'")

        final_args_for_user_script = self.constructor_passed_kwargs.copy()

        if quantization is not None:
            final_args_for_user_script['quantization'] = quantization
        elif 'quantization' not in final_args_for_user_script:
            final_args_for_user_script.pop('quantization', None)

        # 'checkpoint' should be in final_args_for_user_script if it came from YAML via constructor_passed_kwargs
        if 'checkpoint' in final_args_for_user_script and 'model_path' not in final_args_for_user_script:
            final_args_for_user_script['model_path'] = final_args_for_user_script.pop('checkpoint')
            self.logger.debug(f"Mapped 'checkpoint' to 'model_path' for user script.")
        elif 'model_name_hf_id' in final_args_for_user_script and 'model_path' not in final_args_for_user_script:
            final_args_for_user_script['model_path'] = final_args_for_user_script.pop('model_name_hf_id')
            self.logger.debug(f"Mapped 'model_name_hf_id' to 'model_path' for user script.")
        elif 'model_path' not in final_args_for_user_script and 'checkpoint' not in final_args_for_user_script:
             self.logger.warning(f"Neither 'checkpoint' nor 'model_path' found in args for custom script function '{self.function_name}'. The script might rely on a default or fail.")


        if "script_args" in final_args_for_user_script and isinstance(final_args_for_user_script["script_args"], dict):
            user_defined_script_args = final_args_for_user_script.pop("script_args")
            final_args_for_user_script.update(user_defined_script_args)
            self.logger.debug(f"Unpacked 'script_args' into arguments for user script.")

        try:
            module_name = f"custom_model_module.{self.script_path.stem}_{self.function_name}"
            spec = importlib.util.spec_from_file_location(module_name, str(self.script_path))
            
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for model script {str(self.script_path)}")

            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module) # type: ignore[attr-defined]

            if not hasattr(custom_module, self.function_name):
                raise AttributeError(f"Function '{self.function_name}' not found in script '{str(self.script_path)}'")

            load_fn = getattr(custom_module, self.function_name)
            
            self.logger.debug(f"Calling custom model function '{self.function_name}' with effective args: {sorted(list(final_args_for_user_script.keys()))}")
            
            model, tokenizer = load_fn(**final_args_for_user_script)

            if model is None or tokenizer is None:
                raise ValueError(
                    f"Custom model function '{self.function_name}' from script '{self.script_path}' "
                    "did not return a valid (model, tokenizer) tuple. One or both were None."
                )
            
            self.logger.info(f"Custom model '{self.descriptive_name}' and tokenizer loaded successfully from script.")
            return model, tokenizer

        except Exception as e:
            self.logger.error(
                f"Failed to load model/tokenizer from custom script '{str(self.script_path)}' "
                f"using function '{self.function_name}': {e}", exc_info=True
            )
            raise