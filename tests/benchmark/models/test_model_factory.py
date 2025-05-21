import re
import pytest
from unittest.mock import patch, MagicMock
import torch

from src.benchmark.models.models_factory import ModelLoaderFactory #
from src.benchmark.models.concrete_models import ( #
    HuggingFaceModelLoader,
    PyTorchModelLoader,
    TensorFlowModelLoader,
    CustomScriptModelLoader
)
from src.benchmark.models.base_model_loader import BaseModelLoader #

# Simple Factory Tests
def test_model_loader_factory_get_huggingface_loader():
    loader = ModelLoaderFactory.get_model_loader(
        model_name="test-model",
        framework="huggingface"
    )
    assert isinstance(loader, HuggingFaceModelLoader)

def test_model_loader_factory_get_pytorch_loader():
    loader = ModelLoaderFactory.get_model_loader(
        model_name="test-model",
        framework="pytorch"
    )
    assert isinstance(loader, PyTorchModelLoader)

def test_model_loader_factory_get_tensorflow_loader():
    loader = ModelLoaderFactory.get_model_loader(
        model_name="test-model",
        framework="tensorflow"
    )
    assert isinstance(loader, TensorFlowModelLoader)

def test_model_loader_factory_get_custom_script_loader(tmp_path):
    # Temp file creation for the script
    dummy_script_file = tmp_path / "dummy.py"
    dummy_script_file.touch() # Create a dummy script file

    loader = ModelLoaderFactory.get_model_loader(
        model_name="test-custom",
        framework="custom_script",
        model_specific_config_params={
            "script_path": str(dummy_script_file),
            "function_name": "load_fn"
        }
    )
    assert isinstance(loader, CustomScriptModelLoader)
    assert loader.script_path == dummy_script_file

def test_model_loader_factory_unsupported_framework():
    with pytest.raises(ValueError, match="Unsupported framework"):
        ModelLoaderFactory.get_model_loader(model_name="test", framework="unknown_framework")

# More Complex Tests for a Concrete Loader (e.g., HuggingFaceModelLoader)
# These could also be in a separate file like tests/benchmark/models/test_concrete_models.py

@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
@patch('src.benchmark.models.concrete_models.BitsAndBytesConfig')
def test_huggingface_loader_no_quantization(mock_bnb_config, mock_tokenizer_cls, mock_model_lm_cls):
    """Test HuggingFaceModelLoader without quantization."""
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_model_lm_cls.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

    loader = HuggingFaceModelLoader(model_name="gpt2", torch_dtype="float32", offloading=False)
    model, tokenizer = loader.load(quantization=None)

    mock_model_lm_cls.from_pretrained.assert_called_once_with("gpt2", torch_dtype=torch.float32)
    mock_tokenizer_cls.from_pretrained.assert_called_once_with("gpt2")
    assert model is mock_model_instance
    assert tokenizer is mock_tokenizer_instance
    mock_bnb_config.assert_not_called()


@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
@patch('src.benchmark.models.concrete_models.BitsAndBytesConfig')
@patch('torch.cuda.is_bf16_supported', return_value=True) 
@patch('torch.cuda.is_available', return_value=True) 
def test_huggingface_loader_4bit_quantization(mock_cuda_available, mock_bf16_supported, mock_bnb_config_cls, mock_tokenizer_cls, mock_model_lm_cls):
    """Test HuggingFaceModelLoader with 4-bit quantization and bfloat16."""
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_model_lm_cls.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance
    
    mock_bnb_instance = MagicMock()
    mock_bnb_config_cls.return_value = mock_bnb_instance

    loader = HuggingFaceModelLoader(
        model_name="google/gemma-2b",
        torch_dtype="bfloat16", 
        offloading=True 
    )
    model, tokenizer = loader.load(quantization="4bit")

    mock_bnb_config_cls.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True 
    )
    # Check the arguments passed to from_pretrained
    called_args, called_kwargs = mock_model_lm_cls.from_pretrained.call_args
    assert called_args[0] == "google/gemma-2b"
    assert called_kwargs.get("quantization_config") is mock_bnb_instance
    assert called_kwargs.get("device_map") == "auto"
    assert "torch_dtype" not in called_kwargs # Should be handled by bnb_config

    mock_tokenizer_cls.from_pretrained.assert_called_once_with("google/gemma-2b")
    assert model is mock_model_instance
    assert tokenizer is mock_tokenizer_instance


@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
@patch('src.benchmark.models.concrete_models.BitsAndBytesConfig')
@patch('torch.cuda.is_bf16_supported', return_value=False) # Simulate bfloat16 not supported
@patch('torch.cuda.is_available', return_value=True)
def test_huggingface_loader_8bit_quantization_no_bfloat16(mock_cuda_available, mock_bf16_supported, mock_bnb_config_cls, mock_tokenizer_cls, mock_model_lm_cls):
    """Test HuggingFaceModelLoader with 8-bit quantization when bfloat16 is not supported."""
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_model_lm_cls.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance
    
    mock_bnb_instance = MagicMock()
    mock_bnb_config_cls.return_value = mock_bnb_instance

    # torch_dtype is not specified, and bfloat16 is mocked as not supported,
    # so compute_dtype for BNB (if applicable, like in 4bit) would default to float32.
    # For 8-bit, BitsAndBytesConfig primarily cares about `load_in_8bit=True`.
    loader = HuggingFaceModelLoader(model_name="another/model", offloading=False) # torch_dtype will default
    model, tokenizer = loader.load(quantization="8bit")

    # Check that BitsAndBytesConfig was called for 8-bit
    # The first argument to BitsAndBytesConfig() is an object, not direct kwargs in the current transformers version.
    # We check if it was called, and then can inspect the call_args if needed more deeply.
    mock_bnb_config_cls.assert_called_once() 
    # Example to inspect:
    # called_bnb_kwargs = mock_bnb_config_cls.call_args[1] # if kwargs were used
    # assert called_bnb_kwargs['load_in_8bit'] is True 
    # For current BitsAndBytesConfig, it's more like it's initialized and then passed.
    # The loader creates it like: BitsAndBytesConfig(load_in_8bit=True)
    # So we verify that the created config object (mock_bnb_instance) is passed.

    called_args, called_kwargs = mock_model_lm_cls.from_pretrained.call_args
    assert called_args[0] == "another/model"
    assert called_kwargs.get("quantization_config") is mock_bnb_instance
    assert called_kwargs.get("device_map") == "auto" 
    mock_tokenizer_cls.from_pretrained.assert_called_once_with("another/model")


@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
def test_custom_script_model_loader_success(mock_module_from_spec, mock_spec_from_file_location, tmp_path):
    """Test successful loading via CustomScriptModelLoader."""
    script_content = """
import torch
def my_model_loader_fn(model_path, quantization=None, torch_dtype_str=None, custom_param=None, **kwargs):
    # Mock model and tokenizer instances
    mock_model = lambda: None 
    setattr(mock_model, 'name', model_path)
    mock_tokenizer = lambda: None
    setattr(mock_tokenizer, 'name', model_path)
    print(f"Custom model loader called with: {model_path}, quant={quantization}, dtype={torch_dtype_str}, custom_param={custom_param}, other_kwargs={kwargs}")
    if model_path == "fail_load": return None, None # Simulate load failure
    return mock_model, mock_tokenizer
"""
    custom_script_file = tmp_path / "my_model_script.py"
    custom_script_file.write_text(script_content)

    mock_spec = MagicMock()
    mock_spec.loader = MagicMock() # Ensure loader attribute exists
    mock_spec.loader.exec_module = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec
    
    mock_custom_module = MagicMock()
    # A way to execute the script content into the mock module's namespace
    # to make `my_model_loader_fn` available on it.
    namespace = {}
    exec(script_content, namespace)
    mock_custom_module.my_model_loader_fn = namespace['my_model_loader_fn']
    
    mock_module_from_spec.return_value = mock_custom_module

    # These are the kwargs that the factory would prepare and pass to the loader's constructor
    loader_constructor_kwargs = {
        "script_path": str(custom_script_file),
        "function_name": "my_model_loader_fn",
        "checkpoint": "local/my_model_id", # This becomes 'model_path' for the script
        "torch_dtype_str": "bfloat16",    # Argument for the script
        "script_args": {"custom_param": 123} # Nested script_args
    }
    loader = CustomScriptModelLoader(
        model_name="my-custom-model-instance-name", 
        framework="custom_script",
        **loader_constructor_kwargs # Spread the prepared kwargs
    )
    
    # The `load` method of CustomScriptModelLoader will call the script's function
    with patch.object(mock_custom_module, 'my_model_loader_fn', wraps=mock_custom_module.my_model_loader_fn) as wrapped_load_fn:
        model, tokenizer = loader.load(quantization="4bit") 

    wrapped_load_fn.assert_called_once_with(
        model_path="local/my_model_id", # 'checkpoint' from config is mapped to 'model_path' by the loader
        quantization="4bit",           # Passed from load() method
        torch_dtype_str="bfloat16",    # From loader_constructor_kwargs
        custom_param=123               # From script_args, unpacked by the loader
        # any other top-level keys from loader_constructor_kwargs (excluding reserved like script_path, function_name) would also be passed if the script accepted **kwargs
    )
    assert model is not None
    assert tokenizer is not None
    assert getattr(model, 'name', None) == "local/my_model_id" # Check if mock model got attributes


# --- Tests for HuggingFaceModelLoader ---

@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
def test_huggingface_loader_tokenizer_kwargs(mock_tokenizer_cls, mock_model_lm_cls):
    """Test that HuggingFaceModelLoader passes kwargs like trust_remote_code and revision to the tokenizer."""
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_model_lm_cls.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

    loader = HuggingFaceModelLoader(
        model_name="gpt2-medium",
        trust_remote_code=True, # Kwarg for tokenizer
        revision="main"          # Kwarg for tokenizer (and model)
    )
    _, tokenizer = loader.load()

    # Check that AutoTokenizer.from_pretrained is called with these specific kwargs
    mock_tokenizer_cls.from_pretrained.assert_called_with(
        "gpt2-medium",
        trust_remote_code=True,
        revision="main"
    )
    # Also check that model gets them if applicable (from_pretrained for models also accepts revision)
    mock_model_lm_cls.from_pretrained.assert_called_with(
        "gpt2-medium",
        trust_remote_code=True, # The loader passes all its kwargs
        revision="main",
        torch_dtype=torch.bfloat16
    )

@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.is_bf16_supported', return_value=False) # bf16 NOT supported
def test_huggingface_loader_torch_dtype_fallback_to_float32(mock_bf16_not_supported, mock_cuda_available, mock_tokenizer_cls, mock_model_lm_cls):
    """Test torch_dtype falls back to float32 if bfloat16 is requested but not supported."""
    mock_model_lm_cls.from_pretrained.return_value = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

    loader = HuggingFaceModelLoader(model_name="gpt2", torch_dtype="bfloat16") # Request bfloat16
    loader.load()

    # actual_torch_dtype should become float32
    # and this float32 should be passed to from_pretrained (as no quantization is active)
    called_args, called_kwargs = mock_model_lm_cls.from_pretrained.call_args
    assert called_kwargs.get("torch_dtype") == torch.float32

@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.is_bf16_supported', return_value=True) # bf16 IS supported
def test_huggingface_loader_torch_dtype_bfloat16_when_supported_and_no_quant(mock_bf16_supported, mock_cuda_available, mock_tokenizer_cls, mock_model_lm_cls):
    """Test torch_dtype is bfloat16 if not specified, supported, and no quantization."""
    mock_model_lm_cls.from_pretrained.return_value = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

    # torch_dtype is not specified in kwargs, HuggingFaceModelLoader should default to bfloat16 if supported
    loader = HuggingFaceModelLoader(model_name="gpt2")
    loader.load()

    called_args, called_kwargs = mock_model_lm_cls.from_pretrained.call_args
    assert called_kwargs.get("torch_dtype") == torch.bfloat16

@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
@patch('src.benchmark.models.concrete_models.BitsAndBytesConfig')
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.is_bf16_supported', return_value=True)
def test_huggingface_loader_explicit_device_map_with_quantization(mock_bf16_supported, mock_cuda_available, mock_bnb_config_cls, mock_tokenizer_cls, mock_model_lm_cls):
    """Test HuggingFaceModelLoader with an explicit device_map and quantization."""
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_model_lm_cls.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance
    
    mock_bnb_instance = MagicMock()
    mock_bnb_config_cls.return_value = mock_bnb_instance
    
    user_device_map = {"": 0} # Example user-defined device map

    loader = HuggingFaceModelLoader(
        model_name="google/gemma-7b",
        torch_dtype="bfloat16",
        device_map=user_device_map # Explicit device_map
    )
    model, tokenizer = loader.load(quantization="4bit")

    mock_bnb_config_cls.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    called_args, called_kwargs = mock_model_lm_cls.from_pretrained.call_args
    assert called_args[0] == "google/gemma-7b"
    assert called_kwargs.get("quantization_config") is mock_bnb_instance
    assert called_kwargs.get("device_map") == user_device_map # Should use the user-provided one
    assert "torch_dtype" not in called_kwargs

@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
@patch('src.benchmark.models.concrete_models.BitsAndBytesConfig')
def test_huggingface_loader_torch_dtype_ignored_if_bnb_active(mock_bnb_config_cls, mock_tokenizer_cls, mock_model_lm_cls):
    """
    Test that torch_dtype in kwargs is correctly handled (not passed to from_pretrained) 
    when BitsAndBytesConfig is active because bnb_config already specifies compute_dtype.
    """
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_model_lm_cls.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance
    
    mock_bnb_instance = MagicMock()
    mock_bnb_config_cls.return_value = mock_bnb_instance

    loader = HuggingFaceModelLoader(
        model_name="some/model-for-bnb",
        torch_dtype="bfloat16", # This should be used for BNB compute_dtype
        # No offloading, device_map will be auto due to BNB
    )
    loader.load(quantization="4bit")

    mock_bnb_config_cls.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # Correctly used here
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # Crucially, torch_dtype should NOT be in the final_from_pretrained_args
    # if BitsAndBytesConfig is being used, because the compute_dtype is part of bnb_config.
    # The loader has logic: `if quantization_config is None: final_from_pretrained_args["torch_dtype"] = actual_torch_dtype`
    # and `elif "torch_dtype" in final_from_pretrained_args: ... final_from_pretrained_args.pop("torch_dtype")`
    called_args, called_kwargs = mock_model_lm_cls.from_pretrained.call_args
    assert "torch_dtype" not in called_kwargs, "torch_dtype should not be passed to from_pretrained when BitsAndBytesConfig is active"
    assert called_kwargs.get("quantization_config") is mock_bnb_instance
    assert called_kwargs.get("device_map") == "auto" # Default because BNB is active


# --- Tests for PyTorchModelLoader ---

# Mocking PyTorchModelLoader with different quantization scenarios
@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
@patch('src.benchmark.models.concrete_models.BitsAndBytesConfig')
# Mock CUDA and bf16 support as needed for different scenarios
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.is_bf16_supported', return_value=True)
def test_pytorch_loader_no_quantization_bfloat16(mock_bf16_supported, mock_cuda_available, mock_bnb_config_cls, mock_tokenizer_cls, mock_model_lm_cls):
    """Test PyTorchModelLoader without quantization, with bfloat16."""
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_model_lm_cls.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

    loader = PyTorchModelLoader(
        model_name="some/pytorch-model",
        torch_dtype="bfloat16", # Explicitly request bfloat16
        offloading=False,
        trust_remote_code=True # Example of another kwarg
    )
    model, tokenizer = loader.load(quantization=None) # No quantization

    # PyTorchLoader should pass torch_dtype if no BNB quantization
    mock_model_lm_cls.from_pretrained.assert_called_once_with(
        "some/pytorch-model",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    mock_tokenizer_cls.from_pretrained.assert_called_once_with(
        "some/pytorch-model",
        trust_remote_code=True # Assuming PyTorchLoader also passes this to tokenizer
    )
    assert model is mock_model_instance
    assert tokenizer is mock_tokenizer_instance
    mock_bnb_config_cls.assert_not_called()

@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
@patch('src.benchmark.models.concrete_models.BitsAndBytesConfig')
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.is_bf16_supported', return_value=True)
def test_pytorch_loader_4bit_quantization(mock_bf16_supported, mock_cuda_available, mock_bnb_config_cls, mock_tokenizer_cls, mock_model_lm_cls):
    """Test PyTorchModelLoader with 4-bit quantization."""
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_model_lm_cls.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance
    
    mock_bnb_instance = MagicMock()
    mock_bnb_config_cls.return_value = mock_bnb_instance

    loader = PyTorchModelLoader(
        model_name="another/pt-model",
        torch_dtype="bfloat16", # Will be used for bnb_4bit_compute_dtype
        offloading=True         # Should set device_map="auto"
    )
    model, tokenizer = loader.load(quantization="4bit")

    mock_bnb_config_cls.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    called_args, called_kwargs = mock_model_lm_cls.from_pretrained.call_args
    assert called_args[0] == "another/pt-model"
    assert called_kwargs.get("quantization_config") is mock_bnb_instance
    assert called_kwargs.get("device_map") == "auto"
    assert "torch_dtype" not in called_kwargs # Should be handled by bnb_config

    mock_tokenizer_cls.from_pretrained.assert_called_once_with("another/pt-model")
    assert model is mock_model_instance
    assert tokenizer is mock_tokenizer_instance

@patch('src.benchmark.models.concrete_models.AutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
@patch('src.benchmark.models.concrete_models.BitsAndBytesConfig')
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.is_bf16_supported', return_value=False) # bf16 not supported
def test_pytorch_loader_8bit_quantization_no_bfloat16_support(mock_bf16_supported, mock_cuda_available, mock_bnb_config_cls, mock_tokenizer_cls, mock_model_lm_cls):
    """Test PyTorchModelLoader 8-bit, torch_dtype defaults to float32 if bfloat16 not supported/specified."""
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_model_lm_cls.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

    mock_bnb_instance = MagicMock()
    mock_bnb_config_cls.return_value = mock_bnb_instance

    loader = PyTorchModelLoader(
        model_name="pt-model-8bit"
        # No torch_dtype specified, bf16 mocked as not supported => actual_torch_dtype -> float32
        # (or bnb_8bit_compute_dtype if it had such an arg, which it doesn't directly in the same way as 4bit)
    )
    model, tokenizer = loader.load(quantization="8bit")

    mock_bnb_config_cls.assert_called_once()
    # For 8-bit, the call is typically BitsAndBytesConfig(load_in_8bit=True)
    # We can check the arguments if the mock_bnb_config_cls was more specific
    # For now, just checking it was called is a good start for 8-bit.
    # More precise: check that the instance has load_in_8bit=True
    assert mock_bnb_config_cls.call_args[1]['load_in_8bit'] is True


    called_args, called_kwargs = mock_model_lm_cls.from_pretrained.call_args
    assert called_args[0] == "pt-model-8bit"
    assert called_kwargs.get("quantization_config") is mock_bnb_instance
    assert called_kwargs.get("device_map") == "auto" # Default when BNB active

    mock_tokenizer_cls.from_pretrained.assert_called_once_with("pt-model-8bit")


# --- Tests for TensorFlowModelLoader ---

@patch('src.benchmark.models.concrete_models.TFAutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer') # Mock AutoTokenizer as well
def test_tensorflow_loader_basic_load(mock_tokenizer_cls, mock_tf_model_lm_cls):
    """Test TensorFlowModelLoader basic loading scenario."""
    mock_tf_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_tf_model_lm_cls.from_pretrained.return_value = mock_tf_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

    loader = TensorFlowModelLoader(
        model_name="gpt2", # TF gpt2 usually needs from_pt=True if using HF's PT weights
        from_pt=True,      # Explicitly pass relevant TF arg
        trust_remote_code=True
    )
    model, tokenizer = loader.load() # quantization is None by default

    mock_tf_model_lm_cls.from_pretrained.assert_called_once_with(
        "gpt2",
        from_pt=True,
        trust_remote_code=True
    )
    mock_tokenizer_cls.from_pretrained.assert_called_once_with(
        "gpt2",
        trust_remote_code=True # Assuming TF loader also passes this to tokenizer
    )
    assert model is mock_tf_model_instance
    assert tokenizer is mock_tokenizer_instance

@patch('src.benchmark.models.concrete_models.TFAutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
def test_tensorflow_loader_ignores_quantization_and_pytorch_args(mock_tokenizer_cls, mock_tf_model_lm_cls):
    """Test that TensorFlowModelLoader ignores BNB quantization and PyTorch-specific args."""
    mock_tf_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_tf_model_lm_cls.from_pretrained.return_value = mock_tf_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

    loader = TensorFlowModelLoader(
        model_name="some/tf-model",
        # These kwargs should be filtered out or ignored by TFAutoModelForCausalLM
        torch_dtype="bfloat16",
        offloading=True,
        device_map="auto",
        bnb_4bit_quant_type="nf4" # Specific to BNB
    )
    # Pass quantization to load method, which should also be ignored for TF in terms of BNB
    model, tokenizer = loader.load(quantization="4bit")

    # Assert that from_pretrained is called without the PyTorch/BNB specific args
    # TFAutoModelForCausalLM doesn't take torch_dtype, device_map in the same way, or bnb args
    mock_tf_model_lm_cls.from_pretrained.assert_called_once_with(
        "some/tf-model"
        # No 'torch_dtype', 'device_map', 'quantization_config' etc.
        # trust_remote_code might be a default or passed if present in original kwargs
    )
    # Tokenizer call should be simple too
    mock_tokenizer_cls.from_pretrained.assert_called_once_with(
        "some/tf-model"
    )
    assert model is mock_tf_model_instance
    assert tokenizer is mock_tokenizer_instance

@patch('src.benchmark.models.concrete_models.TFAutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
def test_tensorflow_loader_from_pt_retry_logic(mock_tokenizer_cls, mock_tf_model_lm_cls):
    """Test TensorFlowModelLoader's logic to retry with from_pt=True."""
    mock_tf_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()

    # Simulate first call failing as if it's a PyTorch model, then succeeding
    mock_tf_model_lm_cls.from_pretrained.side_effect = [
        OSError("Could not load model. It might be a PyTorch model. Try `from_pt=True`. pytorch_model.bin not found."), # First call fails
        mock_tf_model_instance # Second call (with from_pt=True) succeeds
    ]
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

    loader = TensorFlowModelLoader(
        model_name="a-pytorch-model-on-hub",
        # from_pt is NOT specified initially, to trigger the retry logic
    )
    model, tokenizer = loader.load()

    # Check that TFAutoModelForCausalLM.from_pretrained was called twice
    assert mock_tf_model_lm_cls.from_pretrained.call_count == 2
    
    # Check arguments of the first call (without from_pt)
    first_call_args = mock_tf_model_lm_cls.from_pretrained.call_args_list[0]
    assert first_call_args[0][0] == "a-pytorch-model-on-hub" # model_name
    assert "from_pt" not in first_call_args[1] # from_pt not in kwargs for first call

    # Check arguments of the second call (with from_pt=True)
    second_call_args = mock_tf_model_lm_cls.from_pretrained.call_args_list[1]
    assert second_call_args[0][0] == "a-pytorch-model-on-hub"
    assert second_call_args[1].get("from_pt") is True

    mock_tokenizer_cls.from_pretrained.assert_called_once_with("a-pytorch-model-on-hub")
    assert model is mock_tf_model_instance
    assert tokenizer is mock_tokenizer_instance

@patch('src.benchmark.models.concrete_models.TFAutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
def test_tensorflow_loader_from_pt_specified_no_retry(mock_tokenizer_cls, mock_tf_model_lm_cls):
    """Test that if from_pt is specified, no retry logic is triggered."""
    mock_tf_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_tf_model_lm_cls.from_pretrained.return_value = mock_tf_model_instance
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

    loader = TensorFlowModelLoader(
        model_name="another/tf-model",
        from_pt=False # Explicitly set
    )
    model, tokenizer = loader.load()

    mock_tf_model_lm_cls.from_pretrained.assert_called_once_with(
        "another/tf-model",
        from_pt=False
    )
    assert model is mock_tf_model_instance

@patch('src.benchmark.models.concrete_models.TFAutoModelForCausalLM')
@patch('src.benchmark.models.concrete_models.AutoTokenizer')
def test_tensorflow_loader_general_load_failure(mock_tokenizer_cls, mock_tf_model_lm_cls):
    """Test TensorFlowModelLoader when from_pretrained raises a generic error."""
    mock_tf_model_lm_cls.from_pretrained.side_effect = ValueError("Generic TF loading error")
    mock_tokenizer_cls.from_pretrained.return_value = MagicMock() # Assume tokenizer loads fine

    loader = TensorFlowModelLoader(model_name="failing-tf-model")
    
    with pytest.raises(ValueError, match="Generic TF loading error"):
        loader.load()

    # Ensure it was called (at least once, depending on retry logic if the error was different)
    mock_tf_model_lm_cls.from_pretrained.assert_called()


# --- Additional Tests for CustomScriptModelLoader ---

@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
def test_custom_script_loader_script_raises_exception(mock_module_from_spec, mock_spec_from_file_location, tmp_path):
    """Test CustomScriptModelLoader when the user's script function raises an exception."""
    script_content = """
def my_failing_loader_fn(model_path, **kwargs):
    raise ValueError("Intentional script failure")
"""
    custom_script_file = tmp_path / "my_failing_script.py"
    custom_script_file.write_text(script_content)

    mock_spec = MagicMock()
    mock_spec.loader = MagicMock()
    mock_spec.loader.exec_module = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec
    
    mock_custom_module = MagicMock()
    namespace = {}
    exec(script_content, namespace)
    mock_custom_module.my_failing_loader_fn = namespace['my_failing_loader_fn']
    mock_module_from_spec.return_value = mock_custom_module

    loader_kwargs = {
        "script_path": str(custom_script_file),
        "function_name": "my_failing_loader_fn",
        "checkpoint": "any_model"
    }
    loader = CustomScriptModelLoader(model_name="custom-fail", framework="custom_script", **loader_kwargs)
    
    with pytest.raises(ValueError, match="Intentional script failure"):
        loader.load()

@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
def test_custom_script_loader_script_returns_none(mock_module_from_spec, mock_spec_from_file_location, tmp_path):
    """Test CustomScriptModelLoader when script returns None for model or tokenizer."""
    script_content_model_none = """
def my_loader_model_none(model_path, **kwargs):
    # Returns None for model
    return None, lambda: "tokenizer" 
"""
    script_content_tokenizer_none = """
def my_loader_tokenizer_none(model_path, **kwargs):
    # Returns None for tokenizer
    return lambda: "model", None
"""
    custom_script_model_none = tmp_path / "script_model_none.py"
    custom_script_model_none.write_text(script_content_model_none)

    custom_script_tokenizer_none = tmp_path / "script_tokenizer_none.py"
    custom_script_tokenizer_none.write_text(script_content_tokenizer_none)

    mock_spec = MagicMock()
    mock_spec.loader = MagicMock()
    mock_spec.loader.exec_module = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec
    
    mock_custom_module_model_none = MagicMock()
    namespace_mn = {}
    exec(script_content_model_none, namespace_mn)
    mock_custom_module_model_none.my_loader_model_none = namespace_mn['my_loader_model_none']

    mock_custom_module_tokenizer_none = MagicMock()
    namespace_tn = {}
    exec(script_content_tokenizer_none, namespace_tn)
    mock_custom_module_tokenizer_none.my_loader_tokenizer_none = namespace_tn['my_loader_tokenizer_none']

    # Test case 1: Model is None
    mock_module_from_spec.return_value = mock_custom_module_model_none
    loader_kwargs_model_none = {
        "script_path": str(custom_script_model_none),
        "function_name": "my_loader_model_none",
        "checkpoint": "any_model"
    }
    loader_model_none = CustomScriptModelLoader(model_name="custom-model-none", framework="custom_script", **loader_kwargs_model_none)
    with pytest.raises(ValueError, match=r"Custom model function 'my_loader_model_none' from script .* did not return a valid \(model, tokenizer\) tuple\. One or both were None\."):
        loader_model_none.load()

    # Test case 2: Tokenizer is None
    mock_module_from_spec.return_value = mock_custom_module_tokenizer_none # Change return for next test
    loader_kwargs_tokenizer_none = {
        "script_path": str(custom_script_tokenizer_none),
        "function_name": "my_loader_tokenizer_none",
        "checkpoint": "any_model"
    }
    loader_tokenizer_none = CustomScriptModelLoader(model_name="custom-tokenizer-none", framework="custom_script", **loader_kwargs_tokenizer_none)
    with pytest.raises(ValueError, match=r"Custom model function 'my_loader_tokenizer_none' from script .* did not return a valid \(model, tokenizer\) tuple\. One or both were None\."):
        loader_tokenizer_none.load()


@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
def test_custom_script_loader_module_load_fails(mock_module_from_spec, mock_spec_from_file_location, tmp_path):
    """
    Test CustomScriptModelLoader when the script itself cannot be loaded 
    (e.g., due to SyntaxError or ImportError within the script).
    """
    custom_script_file_spec_none = tmp_path / "broken_script_spec_none.py"
    custom_script_file_spec_none.touch() # <--- CREA IL FILE

    mock_spec_from_file_location.return_value = None
    loader_kwargs_spec_none = {
        "script_path": str(custom_script_file_spec_none),
        "function_name": "any_function_name",
        "checkpoint": "any_model"
    }
    # Simulate that spec_from_file_location returns None
    loader_spec_none = CustomScriptModelLoader(model_name="custom-module-fail-spec-none", framework="custom_script", **loader_kwargs_spec_none)

    expected_match_str_spec_none = f"Could not create module spec for model script {re.escape(str(custom_script_file_spec_none))}"
    with pytest.raises(ImportError, match=expected_match_str_spec_none):
        loader_spec_none.load()

    # --- Part 2: Simulate exec_module failing ---
    custom_script_file_exec_fail = tmp_path / "broken_script_exec_fail.py"
    custom_script_file_exec_fail.touch() # Create the file

    mock_spec_exec_fail = MagicMock()
    mock_spec_exec_fail.loader = MagicMock()
    mock_spec_exec_fail.loader.exec_module.side_effect = SyntaxError("Bad syntax in custom script")
    mock_spec_from_file_location.return_value = mock_spec_exec_fail
    mock_module_from_spec.return_value = MagicMock()


    loader_kwargs_exec_fail = {
        "script_path": str(custom_script_file_exec_fail),
        "function_name": "any_function_name",
        "checkpoint": "any_model"
    }
    loader_exec_fail = CustomScriptModelLoader(model_name="custom-module-fail-exec", framework="custom_script", **loader_kwargs_exec_fail)

    with pytest.raises(SyntaxError, match="Bad syntax in custom script"):
        loader_exec_fail.load()


@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
def test_custom_script_loader_function_not_found_in_script(mock_module_from_spec, mock_spec_from_file_location, tmp_path):
    """Test CustomScriptModelLoader when the specified function_name is not in the loaded script."""
    script_content = """
def some_other_function(model_path, **kwargs):
    return "model", "tokenizer"
"""
    custom_script_file = tmp_path / "script_missing_func.py"
    custom_script_file.write_text(script_content)
    custom_script_file.touch()

    mock_spec = MagicMock()
    mock_spec.loader = MagicMock()
    mock_spec = MagicMock()
    mock_spec.loader = MagicMock()

    # This is a mock module that will be returned by module_from_spec
    class ModuleLikeObject:
        pass
    actual_custom_module = ModuleLikeObject()
    mock_module_from_spec.return_value = actual_custom_module

    # This is a mock for the exec_module method
    def exec_module_effect(module_obj): # This simulates the exec_module behavior
        exec(script_content, module_obj.__dict__)
    mock_spec.loader.exec_module.side_effect = exec_module_effect
    mock_spec_from_file_location.return_value = mock_spec

    loader_kwargs = {
        "script_path": str(custom_script_file),
        "function_name": "target_function_name", # This function does not exist in the script
        "checkpoint": "any_model"
    }
    loader = CustomScriptModelLoader(model_name="custom-func-not-found", framework="custom_script", **loader_kwargs)

    # The loader will try to call the function that doesn't exist
    expected_match_str_attr = f"Function 'target_function_name' not found in script '{re.escape(str(custom_script_file))}'"
    with pytest.raises(AttributeError, match=expected_match_str_attr):
        loader.load()
