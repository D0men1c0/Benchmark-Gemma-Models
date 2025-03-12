import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Class for loading models with support for flexible quantization and offloading.

    :param checkpoint: Path or identifier for the model.
    :param framework: Framework to use for loading the model (e.g., "huggingface", "pytorch", "ggml", "safetensors").
    :param quantization: Optional quantization type (e.g., "4bit", "8bit", "nf4", "int8").
    :param offloading: Whether to enable offloading to CPU if VRAM is insufficient.
    """

    def __init__(self, checkpoint: str, framework: str, quantization: Optional[str] = None, offloading: bool = False):
        self.checkpoint = checkpoint
        self.framework = framework.lower()
        self.quantization = quantization
        self.offloading = offloading

    def load(self) -> Dict[str, Any]:
        """
        Loads the model and tokenizer from the checkpoint with desired configurations.

        :return: A dictionary with the model and tokenizer.
        """
        supported_frameworks = ["huggingface", "pytorch", "ggml", "safetensors"]
        if self.framework not in supported_frameworks:
            raise NotImplementedError(f"Framework {self.framework} not supported yet. "
                                     f"Supported frameworks: {', '.join(supported_frameworks)}")

        try:
            # Flexible quantization support
            quantization_config = {}
            if self.quantization:
                if self.quantization in ["4bit", "8bit"]:
                    quantization_config = {f"load_in_{self.quantization}": True}
                elif self.quantization == "nf4":
                    quantization_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}
                elif self.quantization == "int8":
                    quantization_config = {"load_in_8bit": True}
                else:
                    raise ValueError(f"Unsupported quantization type: {self.quantization}")

            # Model loading with framework selection
            if self.framework == "huggingface":
                model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint,
                    device_map="auto" if self.offloading else None,
                    **quantization_config
                )
                tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
                
            elif self.framework == "pytorch":
                logger.info(f"Loading PyTorch model from {self.checkpoint}")
                model = torch.load(self.checkpoint)
                
                # For PyTorch, try to load separate tokenizer or extract from dict
                if isinstance(model, dict) and "model" in model and "tokenizer" in model:
                    tokenizer = model["tokenizer"]
                    model = model["model"]
                else:
                    tokenizer_path = self.checkpoint.replace(".pt", "_tokenizer")
                    if not tokenizer_path.endswith("_tokenizer"):
                        tokenizer_path += "_tokenizer"
                    try:
                        tokenizer = torch.load(tokenizer_path)
                    except FileNotFoundError:
                        logger.warning(f"Tokenizer not found at {tokenizer_path}")
                        tokenizer = None
                        
            elif self.framework == "ggml":
                try:
                    from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
                except ImportError:
                    raise ImportError("Loading GGML models requires ctransformers library. "
                                    "Install with: pip install ctransformers")
                    
                logger.info(f"Loading GGML model from {self.checkpoint}")
                model = CTAutoModelForCausalLM.from_pretrained(
                    self.checkpoint,
                    model_type="llama"  # This may need to be a parameter
                )
                
                # Try to load compatible tokenizer
                tokenizer_name = self.checkpoint.split("/")[-1].split("-ggml")[0]
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                except Exception:
                    logger.warning("Could not load tokenizer for GGML model")
                    tokenizer = None
                    
            elif self.framework == "safetensors":
                try:
                    from safetensors.torch import load_file
                    from transformers import AutoConfig
                except ImportError:
                    raise ImportError("Loading safetensors models requires safetensors library. "
                                    "Install with: pip install safetensors")
                    
                logger.info(f"Loading safetensors model from {self.checkpoint}")
                state_dict = load_file(self.checkpoint)
                
                # Load config and initialize model
                config_path = self.checkpoint.replace(".safetensors", "_config.json")
                config = AutoConfig.from_pretrained(config_path)
                model = AutoModelForCausalLM.from_config(config)
                model.load_state_dict(state_dict)
                
                # Try to load tokenizer
                tokenizer_path = self.checkpoint.replace(".safetensors", "_tokenizer")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                except Exception:
                    logger.warning(f"Tokenizer not found at {tokenizer_path}")
                    tokenizer = None

            return {"model": model, "tokenizer": tokenizer}

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and self.framework == "huggingface":
                logger.warning(f"Model {self.checkpoint} could not fit in GPU memory. Trying with CPU offloading.")
                model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint,
                    device_map="cpu",
                    **quantization_config
                )
                tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
                return {"model": model, "tokenizer": tokenizer}
            else:
                raise