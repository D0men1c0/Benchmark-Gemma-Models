import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_local_hf_model_example(
    model_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    quantization: Optional[str] = None,
    torch_dtype_str: Optional[str] = None,
    device_map: Optional[str] = "auto",
    model_load_args: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Tuple[Any, Any]:
    """
    Example function to load a Hugging Face compatible model and tokenizer from a local path.
    This function will be called by CustomScriptModelLoader.

    :param model_path: Path to the model directory or Hugging Face model name.
    :param tokenizer_name_or_path: Path or name of the tokenizer. If None, defaults to model_path.
    :param quantization: Quantization type (e.g., "4bit", "8bit"). Defaults to None.
    :param torch_dtype_str: Torch dtype string (e.g., "float16", "bfloat16"). Defaults to None.
    :param device_map: Device map for model loading. Defaults to "auto".
    :param model_load_args: Additional arguments for model loading.
    :param kwargs: Additional arguments for the function.
    :return: A tuple of (model, tokenizer).
    """
    logger.info(f"Custom model loading function 'load_local_hf_model_example' called.")
    logger.info(f"  Model Path: {model_path}")
    logger.info(f"  Tokenizer Path/Name: {tokenizer_name_or_path or model_path}")
    logger.info(f"  Quantization: {quantization}")
    logger.info(f"  Torch Dtype: {torch_dtype_str}")
    logger.info(f"  Device Map: {device_map}")
    logger.info(f"  Model Load Args: {model_load_args}")
    logger.info(f"  Other Script Args: {kwargs}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    actual_tokenizer_path = tokenizer_name_or_path if tokenizer_name_or_path else model_path

    # --- Tokenizer Loading ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(actual_tokenizer_path, trust_remote_code=True)
        logger.info(f"Tokenizer loaded from: {actual_tokenizer_path}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {actual_tokenizer_path}: {e}")
        raise

    # --- Model Loading Parameters ---
    effective_model_load_args = model_load_args.copy() if model_load_args else {}

    # 1. Determine torch_dtype
    actual_torch_dtype = torch.float32 # Default
    if torch_dtype_str:
        if torch_dtype_str.lower() == "bfloat16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                actual_torch_dtype = torch.bfloat16
            else:
                logger.warning("Warning: bfloat16 requested but not supported, defaulting to float32.")
        elif torch_dtype_str.lower() == "float16":
            actual_torch_dtype = torch.float16
        elif torch_dtype_str.lower() == "float32":
            actual_torch_dtype = torch.float32
        else:
            logger.warning(f"Warning: Unrecognized torch_dtype_str '{torch_dtype_str}', defaulting to float32.")
    
    # Only set torch_dtype in args if not using bitsandbytes or if it's for compute_dtype
    if quantization not in ["4bit", "8bit"] or (quantization == "4bit" and actual_torch_dtype):
         if actual_torch_dtype: # Ensure it's not None if we are setting it
            effective_model_load_args["torch_dtype"] = actual_torch_dtype


    # 2. Handle Quantization with BitsAndBytesConfig
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=actual_torch_dtype, # Crucial for 4-bit
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        effective_model_load_args["quantization_config"] = bnb_config
        effective_model_load_args["device_map"] = device_map # Bitsandbytes needs device_map
        logger.info(f"Configured for 4-bit quantization with compute dtype: {actual_torch_dtype}, device_map: {device_map}")
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        effective_model_load_args["quantization_config"] = bnb_config
        effective_model_load_args["device_map"] = device_map # Bitsandbytes needs device_map
        logger.info(f"Configured for 8-bit quantization, device_map: {device_map}")
    elif quantization is None:
        # If not using bitsandbytes quantization, device_map is still good for large models
        if "device_map" not in effective_model_load_args: # Respect if user passed it in model_load_args
            effective_model_load_args["device_map"] = device_map
        logger.info(f"No bitsandbytes quantization. Device_map: {effective_model_load_args.get('device_map')}")
    else:
        logger.warning(f"Warning: Quantization type '{quantization}' not explicitly handled by this script's bitsandbytes setup.")

    # --- Model Loading ---
    try:
        logger.info(f"Loading model from '{model_path}' with args: {effective_model_load_args}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **effective_model_load_args
        )
        logger.info(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

    return model, tokenizer