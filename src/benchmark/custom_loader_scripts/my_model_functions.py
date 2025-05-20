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

    is_local_path = Path(model_path).is_dir() # Controlla se è una directory esistente
    if is_local_path:
        logger.info(f"  Interpreting '{model_path}' as a local directory path.")
    else:
        logger.info(f"  Interpreting '{model_path}' as a Hugging Face Hub model ID (or a path that will be resolved by HF).")

    actual_tokenizer_path = tokenizer_name_or_path if tokenizer_name_or_path else model_path

    try:
        tokenizer_trust_remote_code = model_load_args.get("trust_remote_code", True) if model_load_args else True
        tokenizer = AutoTokenizer.from_pretrained(actual_tokenizer_path, trust_remote_code=tokenizer_trust_remote_code)
        logger.info(f"Tokenizer loaded from: {actual_tokenizer_path}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {actual_tokenizer_path}: {e}")
        raise

    effective_model_load_args = model_load_args.copy() if model_load_args else {}
    if "trust_remote_code" not in effective_model_load_args:
         effective_model_load_args["trust_remote_code"] = True


    actual_torch_dtype = torch.float32
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
    
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=actual_torch_dtype,
            bnb_4bit_quant_type=effective_model_load_args.pop("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=effective_model_load_args.pop("bnb_4bit_use_double_quant", True),
        )
        effective_model_load_args["quantization_config"] = bnb_config
        if "device_map" not in effective_model_load_args:
             effective_model_load_args["device_map"] = device_map 
        logger.info(f"Configured for 4-bit quantization. Compute dtype: {actual_torch_dtype}, device_map: {effective_model_load_args['device_map']}")
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        effective_model_load_args["quantization_config"] = bnb_config
        if "device_map" not in effective_model_load_args:
             effective_model_load_args["device_map"] = device_map
        logger.info(f"Configured for 8-bit quantization. device_map: {effective_model_load_args['device_map']}")
    elif quantization is None:
        if "device_map" not in effective_model_load_args:
            effective_model_load_args["device_map"] = device_map
        if actual_torch_dtype != torch.float32 or torch_dtype_str == "float32":
            effective_model_load_args["torch_dtype"] = actual_torch_dtype
        logger.info(f"No bitsandbytes quantization. device_map: {effective_model_load_args.get('device_map')}, torch_dtype: {effective_model_load_args.get('torch_dtype')}")
    else:
        logger.warning(f"Warning: Quantization type '{quantization}' not explicitly handled by this script's bitsandbytes setup.")
        if "device_map" not in effective_model_load_args: # Fallback device_map
            effective_model_load_args["device_map"] = device_map

    try:
        logger.info(f"Loading model from '{model_path}' with args: {effective_model_load_args}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **effective_model_load_args
        )
        logger.info(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

    return model, tokenizer