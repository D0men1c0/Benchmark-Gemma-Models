import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import torch
import tensorflow as tf # Import tensorflow for TF-specific tests
from datasets import Dataset, IterableDataset 

from src.benchmark.benchmark_loader import BenchmarkRunner # Already present
from src.config_models import ( # Already present
    BenchmarkConfig, GeneralConfig, ModelConfig, TaskConfig,
    DatasetConfig, MetricConfig, ReportingConfig, EvaluationConfig,
    AdvancedConfig, ModelParamsConfig
)

from src.benchmark.tasks.concrete_task_handlers import (
    GlueClassificationPromptingTaskHandler,
    GlueTextPairPromptingTaskHandler,
    CustomScriptTaskHandler,
    MultipleChoiceQATaskHandler,
    SummarizationTaskHandler,
    MathReasoningGenerationTaskHandler,
    TranslationTaskHandler
)
from src.benchmark.tasks.base_task_handler import TaskHandler
from src.benchmark.prompting.prompt_builder_factory import PromptBuilderFactory


class ConcreteTestTaskHandler(TaskHandler):
    def process_batch(self, batch: dict) -> tuple[list, list]:
        # Dummy implementation, can be simple for most BaseTaskHandler tests
        # For tests that rely on specific output from process_batch, this might need adjustment
        # or be mocked directly on the instance.
        return ([], []) 

# --- Fixtures (ensure all necessary fixtures from your previous file are present) ---
@pytest.fixture
def minimal_general_config():
    return GeneralConfig(output_dir=Path("test_benchmark_outputs"), experiment_name="test_exp")

@pytest.fixture
def minimal_reporting_config():
    return ReportingConfig(format="json", output_dir=Path("test_reports"))

@pytest.fixture
def minimal_advanced_config():
    return AdvancedConfig(batch_size=2, max_new_tokens=10, generate_max_length=10) # ensure generate_max_length is consistent

@pytest.fixture
def minimal_evaluation_config():
    return EvaluationConfig(log_interval=1)

@pytest.fixture
def minimal_model_params_config():
    return ModelParamsConfig()

@pytest.fixture
def dummy_model_config_payload():
    return {"name": "dummy-model", "framework": "huggingface", "checkpoint": "dummy/model-checkpoint"}

@pytest.fixture
def dummy_model_config(dummy_model_config_payload):
    return ModelConfig(**dummy_model_config_payload)

@pytest.fixture
def dummy_dataset_config_payload():
    return {"name": "dummy-dataset", "source_type": "hf_hub", "split": "train", "max_samples": 10}

@pytest.fixture
def dummy_dataset_config(dummy_dataset_config_payload):
    return DatasetConfig(**dummy_dataset_config_payload)

@pytest.fixture
def dummy_metric_config_payload():
    return {"name": "accuracy"}

@pytest.fixture
def dummy_metric_config(dummy_metric_config_payload):
    return MetricConfig(**dummy_metric_config_payload)

@pytest.fixture
def dummy_task_config_obj(dummy_dataset_config, dummy_metric_config):
    return TaskConfig(
        name="dummy-task", type="classification", datasets=[dummy_dataset_config],
        evaluation_metrics=[dummy_metric_config],
        handler_options={"prompt_builder_type": "default"}
    )

@pytest.fixture
def alt_dataset_config_payload():
    return {"name": "alt-dummy-dataset", "source_type": "hf_hub", "split": "test", "max_samples": 5}

@pytest.fixture
def alt_dataset_config(alt_dataset_config_payload):
    return DatasetConfig(**alt_dataset_config_payload)

@pytest.fixture
def alt_metric_config_payload():
    return {"name": "f1_score"}

@pytest.fixture
def alt_metric_config(alt_metric_config_payload):
    return MetricConfig(**alt_metric_config_payload)

@pytest.fixture
def dummy_task_config_obj_alt(alt_dataset_config, alt_metric_config):
    return TaskConfig(
        name="alt-dummy-task", 
        type="classification", 
        datasets=[alt_dataset_config],
        evaluation_metrics=[alt_metric_config],
        handler_options={"prompt_builder_type": "default"}
    )

@pytest.fixture
def dummy_model_config_alt(dummy_model_config_payload):
    alt_model_payload = dummy_model_config_payload.copy()
    alt_model_payload["name"] = "alt-dummy-model"
    alt_model_payload["checkpoint"] = "dummy/alt-model-checkpoint"
    return ModelConfig(**alt_model_payload)

@pytest.fixture
def basic_benchmark_config(
    minimal_general_config, dummy_task_config_obj, dummy_model_config,
    minimal_model_params_config, minimal_evaluation_config,
    minimal_reporting_config, minimal_advanced_config
):
    return BenchmarkConfig(
        general=minimal_general_config, tasks=[dummy_task_config_obj], models=[dummy_model_config],
        model_parameters=minimal_model_params_config, evaluation=minimal_evaluation_config,
        reporting=minimal_reporting_config, advanced=minimal_advanced_config
    )

# --- Fixtures for TaskHandler Tests ---

@pytest.fixture
def mock_pytorch_model():
    model = MagicMock(spec=torch.nn.Module) 
    model.hf_device_map = None 
    model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(1))])) 
    model.config = MagicMock(pad_token_id=None, is_encoder_decoder=False)
    model.generate = MagicMock(return_value=torch.tensor([[0, 1, 2, 3, 4]])) 
    model.device = torch.device("cpu") 
    return model

@pytest.fixture
def mock_tensorflow_model():
    model = MagicMock()
    model.__class__.__name__ = "TFForCausalLM" 
    model.generate = MagicMock(return_value=tf.constant([[0, 1, 2, 3, 4]]))
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.padding_side = "right" 
    tokenizer.name_or_path = "mock_tokenizer_name"

    mock_batch_encoding_object = MagicMock(name="MockBatchEncoding")

    default_tokenized_output_dict = {
        "input_ids": torch.tensor([[0, 1, 2]]), 
        "attention_mask": torch.tensor([[1, 1, 1]])
    }

    def to_side_effect(device_arg):
        return {key: value.clone().detach()
                for key, value in mock_batch_encoding_object._data_for_to.items()}

    mock_batch_encoding_object.to = MagicMock(side_effect=to_side_effect)
    mock_batch_encoding_object._data_for_to = default_tokenized_output_dict 

    tokenizer.return_value = mock_batch_encoding_object

    tokenizer.batch_decode = MagicMock(
        side_effect=lambda token_ids_list, skip_special_tokens, clean_up_tokenization_spaces: 
            [f"decoded_seq_{i}" for i in range(len(token_ids_list))]
    )
    return tokenizer

@pytest.fixture
def base_task_handler_pytorch(mock_pytorch_model, mock_tokenizer, minimal_advanced_config):
    with patch('src.benchmark.tasks.base_task_handler.setup_logger') as mock_task_setup_logger: # Correct patch path
        mock_logger_instance = MagicMock()
        mock_task_setup_logger.return_value = mock_logger_instance
        handler_args = minimal_advanced_config.model_dump(exclude_none=True)
        handler_args.setdefault("max_new_tokens", 10)
        handler_args.setdefault("generate_max_length", 10)
        handler_args.setdefault("tokenizer_max_length", handler_args["generate_max_length"])

        # Use the concrete test handler
        handler = ConcreteTestTaskHandler(model=mock_pytorch_model, tokenizer=mock_tokenizer, device="cpu", advanced_args=handler_args)
        handler.logger = mock_logger_instance 
        return handler

@pytest.fixture
def base_task_handler_tensorflow(mock_tensorflow_model, mock_tokenizer, minimal_advanced_config):
    with patch('src.benchmark.tasks.base_task_handler.setup_logger') as mock_task_setup_logger:
        mock_logger_instance = MagicMock()
        mock_task_setup_logger.return_value = mock_logger_instance
        handler_args = minimal_advanced_config.model_dump(exclude_none=True)
        handler_args.setdefault("max_new_tokens", 10)
        handler_args.setdefault("generate_max_length", 10)
        handler_args.setdefault("tokenizer_max_length", handler_args["generate_max_length"])

        handler = ConcreteTestTaskHandler(model=mock_tensorflow_model, tokenizer=mock_tokenizer, device="cpu", advanced_args=handler_args)
        handler.logger = mock_logger_instance
        return handler

@pytest.fixture
def mcqa_handler_args(minimal_advanced_config, dummy_task_config_obj):
    args = minimal_advanced_config.model_dump(exclude_none=True)
    if dummy_task_config_obj.handler_options:
        args.update(dummy_task_config_obj.handler_options)
    # Add dataset specific info that _run_task_evaluation would pass
    if dummy_task_config_obj.datasets: # Check if datasets list is not empty
        args["dataset_name"] = dummy_task_config_obj.datasets[0].name
        args["dataset_config_name"] = dummy_task_config_obj.datasets[0].config
    else: # Provide defaults if no datasets are in the task config
        args["dataset_name"] = "default_dataset_name_for_mcqa"
        args["dataset_config_name"] = None
    return args

# --- Fixture for GLUE SST-2 Handler Args ---
@pytest.fixture
def glue_sst2_handler_args(minimal_advanced_config):
    args = minimal_advanced_config.model_dump(exclude_none=True)
    args.update({
        "prompt_builder_type": "glue_sst2_prompt", # Specific to SST-2
        "postprocessor_key": "glue_sst2_output",   # Specific to SST-2
        "dataset_config_name": "sst2", # Context for prompt builder/postprocessor
        "dataset_name": "glue",
        # "prompt_template": "Custom SST-2 template: {input_text}" # Optionally override default
    })
    return args

# --- Fixture for GLUE MRPC Handler Args ---
@pytest.fixture
def glue_mrpc_handler_args(minimal_advanced_config):
    args = minimal_advanced_config.model_dump(exclude_none=True)
    args.update({
        "prompt_builder_type": "glue_mrpc_prompt", # Specific to MRPC
        "postprocessor_key": "glue_mrpc_output",   # Specific to MRPC
        "dataset_config_name": "mrpc", # Context for prompt builder/postprocessor
        "dataset_name": "glue",
    })
    return args
    
# --- Fixture for CustomScriptTaskHandler Args ---
@pytest.fixture
def custom_script_handler_args(minimal_advanced_config, tmp_path):
    args = minimal_advanced_config.model_dump(exclude_none=True)
    
    # Create a dummy handler script
    dummy_script_content = """
from typing import Dict, Any, Tuple, List
def my_custom_batch_processor(batch: Dict[str, Any], model: Any, tokenizer: Any, device: str, advanced_args: Dict[str, Any], script_args: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
    prompts = [f"Custom prompt for: {item}" for item in batch.get("input_text", [])]
    labels = batch.get("label_index", [])
    return prompts, labels
"""
    script_file = tmp_path / "dummy_handler_script.py"
    script_file.write_text(dummy_script_content)
    
    args.update({
        "handler_script_path": str(script_file),
        "handler_function_name": "my_custom_batch_processor",
        "handler_script_args": {"custom_arg_for_script": "value123"},
        "postprocessor_key": "generic_generation" # Or mock PostProcessorFactory directly
    })
    return args


# --- Tests for BaseTaskHandler ---

@patch('src.benchmark.tasks.base_task_handler.setup_logger') # For logger inside TaskHandler
def test_base_task_handler_pytorch_init_decoder_only_padding_side(
    mock_task_setup_logger, mock_pytorch_model, mock_tokenizer, minimal_advanced_config
):
    mock_task_setup_logger.return_value = MagicMock()
    mock_pytorch_model.config.is_encoder_decoder = False # Ensure it's decoder-only
    mock_tokenizer.padding_side = "right" # Start with right padding

    handler_args = minimal_advanced_config.model_dump(exclude_none=True)
    handler = ConcreteTestTaskHandler(model=mock_pytorch_model, tokenizer=mock_tokenizer, device="cpu", advanced_args=handler_args)
    
    assert handler.tokenizer.padding_side == "left"
    handler.logger.debug.assert_any_call("PyTorch Decoder-only model: Setting tokenizer padding_side to 'left'.")

@patch('src.benchmark.tasks.base_task_handler.setup_logger')
def test_base_task_handler_tokenizer_pad_token_setup(mock_task_setup_logger, mock_pytorch_model, mock_tokenizer, minimal_advanced_config):
    mock_task_setup_logger.return_value = MagicMock()

    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos_test>"
    mock_tokenizer.eos_token_id = 50257
    mock_tokenizer.pad_token_id = 50257 

    handler_args = minimal_advanced_config.model_dump(exclude_none=True)
    handler = ConcreteTestTaskHandler(model=mock_pytorch_model, tokenizer=mock_tokenizer, device="cpu", advanced_args=handler_args)

    assert handler.tokenizer.pad_token == "<eos_test>"
    handler.logger.warning.assert_any_call(
        f"Tokenizer for '{getattr(mock_tokenizer, 'name_or_path', 'unknown')}' has no pad_token. "
        f"Setting pad_token to eos_token ('<eos_test>')."
    )
    assert handler.model.config.pad_token_id == 50257


def test_base_task_handler_generate_text_pytorch(base_task_handler_pytorch):
    import torch

    prompts = ["prompt1", "prompt2"]

    input_ids_tensor = torch.tensor([[0, 1, 2], [0, 1, 3]])
    attention_mask_tensor = torch.tensor([[1, 1, 1], [1, 1, 1]])

    expected_prepared_inputs_dict = {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor
    }

    # Mock the tokenizer to return input tensors
    tokenizer_output_mock = base_task_handler_pytorch.tokenizer.return_value
    tokenizer_output_mock._data_for_to = expected_prepared_inputs_dict
    tokenizer_output_mock.to.return_value = expected_prepared_inputs_dict

    # Mock the model's generate output
    base_task_handler_pytorch.model.generate.return_value = torch.tensor([
        [0, 1, 2, 10, 11],
        [0, 1, 3, 12, 13]
    ])

    # Define a mock batch decode side effect
    def mock_batch_decode_side_effect(token_ids_list, skip_special_tokens, clean_up_tokenization_spaces):
        decoded_texts = []
        for tokens in token_ids_list:
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            if tokens[-2:] == [10, 11]:
                decoded_texts.append("decoded_new1")
            elif tokens[-2:] == [12, 13]:
                decoded_texts.append("decoded_new2")
            else:
                decoded_texts.append(f"unexpected_tokens_{tokens}")
        return decoded_texts

    base_task_handler_pytorch.tokenizer.batch_decode.side_effect = mock_batch_decode_side_effect

    # Run the method under test
    generated_texts = base_task_handler_pytorch._generate_text(prompts)

    # Assert decoded output
    assert generated_texts == ["decoded_new1", "decoded_new2"]

    # Verify tokenizer was called correctly
    base_task_handler_pytorch.tokenizer.assert_called_once_with(
        prompts,
        return_tensors="pt",
        padding=base_task_handler_pytorch.padding,
        truncation=base_task_handler_pytorch.truncation,
        max_length=base_task_handler_pytorch.tokenizer_max_length,
        add_special_tokens=True
    )

    # Ensure .to() was called on tokenizer output
    tokenizer_output_mock.to.assert_called_once_with(base_task_handler_pytorch.device)

    # Extract the actual call args from the mock
    called_args, called_kwargs = base_task_handler_pytorch.model.generate.call_args

    # Compare tensors with torch.equal
    assert torch.equal(called_kwargs["input_ids"], input_ids_tensor)
    assert torch.equal(called_kwargs["attention_mask"], attention_mask_tensor)

    # Compare the remaining parameters
    assert called_kwargs["max_new_tokens"] == base_task_handler_pytorch.max_new_tokens
    assert called_kwargs["num_beams"] == base_task_handler_pytorch.num_beams
    assert called_kwargs["do_sample"] == base_task_handler_pytorch.do_sample
    assert called_kwargs["pad_token_id"] == base_task_handler_pytorch.tokenizer.pad_token_id
    assert called_kwargs["use_cache"] == base_task_handler_pytorch.use_cache


# --- Tests for MultipleChoiceQATaskHandler ---

@pytest.fixture
def mcqa_handler_args(minimal_advanced_config, dummy_task_config_obj):
    args = minimal_advanced_config.model_dump(exclude_none=True)
    if dummy_task_config_obj.handler_options:
        args.update(dummy_task_config_obj.handler_options)
    # Add dataset specific info that _run_task_evaluation would pass
    args["dataset_name"] = dummy_task_config_obj.datasets[0].name
    args["dataset_config_name"] = dummy_task_config_obj.datasets[0].config
    return args

@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.PostProcessorFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.TaskHandler._generate_text') # Mock the base class's generate
@patch('src.benchmark.tasks.base_task_handler.setup_logger') # For logger inside Concrete Handler
def test_mcqa_handler_process_batch_normal_case(
    mock_setup_logger_ch, mock_generate_text, mock_post_processor_factory, 
    mock_prompt_builder_factory, mock_pytorch_model, mock_tokenizer, mcqa_handler_args
):
    mock_setup_logger_ch.return_value = MagicMock()
    mock_prompt_builder_instance = MagicMock()
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance
    mock_post_processor_instance = MagicMock()
    mock_post_processor_factory.get_processor.return_value = mock_post_processor_instance

    handler = MultipleChoiceQATaskHandler(
        model=mock_pytorch_model, 
        tokenizer=mock_tokenizer, 
        device="cpu", 
        advanced_args=mcqa_handler_args
    )
    handler.logger = MagicMock() # Override logger for handler instance

    batch_data = {
        "question": ["Q1?", "Q2?"],
        "choices": [["A1","B1","C1","D1"], ["A2","B2","C2","D2"]],
        "label_index": [0, 1],
        "subject": ["Sub1", "Sub2"]
    }
    
    expected_prompts = ["Formatted Q1", "Formatted Q2"]
    mock_prompt_builder_instance.build_prompts.return_value = expected_prompts
    
    raw_model_outputs = ["Raw Answer 1", "Raw Answer 2"]
    mock_generate_text.return_value = raw_model_outputs
    
    processed_preds = ["A", "B"]
    processed_labels = ["A", "B"] # Assuming postprocessor converts labels
    mock_post_processor_instance.process.return_value = (processed_preds, processed_labels)

    final_preds, final_labels = handler.process_batch(batch_data)

    assert final_preds == processed_preds
    assert final_labels == processed_labels

    # Verify PromptBuilder was called correctly
    expected_prompt_builder_items = [
        {
            "question": "Q1?", 
            "choices_formatted_str": "A) A1\nB) B1\nC) C1\nD) D1", 
            "subject": "Sub1",
            "choice_A": "A1", "choice_B": "B1", "choice_C": "C1", "choice_D": "D1"
        },
        {
            "question": "Q2?", 
            "choices_formatted_str": "A) A2\nB) B2\nC) C2\nD) D2", 
            "subject": "Sub2",
            "choice_A": "A2", "choice_B": "B2", "choice_C": "C2", "choice_D": "D2"
        }
    ]
    mock_prompt_builder_instance.build_prompts.assert_called_once_with(expected_prompt_builder_items)
    
    # Verify _generate_text was called
    mock_generate_text.assert_called_once_with(expected_prompts)
    
    # Verify PostProcessor was called
    mock_post_processor_factory.get_processor.assert_called_once_with("mmlu_generation") # Default for MCQA
    mock_post_processor_instance.process.assert_called_once_with(raw_model_outputs, [0, 1], batch_data)


@patch('src.benchmark.tasks.base_task_handler.setup_logger') # For logger inside Concrete Handler
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
def test_mcqa_handler_process_batch_no_questions(mock_setup_logger_ch, mock_prompt_builder_factory, mock_pytorch_model, mock_tokenizer, mcqa_handler_args):
    mock_setup_logger_ch.return_value = MagicMock()
    mock_prompt_builder_instance = MagicMock()
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance

    handler = MultipleChoiceQATaskHandler(
        model=mock_pytorch_model, 
        tokenizer=mock_tokenizer, 
        device="cpu", 
        advanced_args=mcqa_handler_args
    )
    handler.logger = MagicMock()

    batch_data_no_q = {
        "question": [], # No questions
        "choices": [],
        "label_index": [],
        "subject": []
    }
    preds, labels = handler.process_batch(batch_data_no_q)
    assert preds == []
    assert labels == []
    handler.logger.warning.assert_called_with(
        "MCQA batch: Question count (0) and Label count (0) mismatch, or no questions. Skipping batch."
    )

@patch('src.benchmark.tasks.base_task_handler.setup_logger')
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
def test_mcqa_handler_process_batch_no_choices_and_no_template(
    mock_setup_logger_ch, mock_prompt_builder_factory, mock_pytorch_model, mock_tokenizer, mcqa_handler_args
):
    mock_setup_logger_ch.return_value = MagicMock()
    mock_prompt_builder_instance = MagicMock()
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance

    # Modify args to remove prompt_template to hit the specific warning
    args_no_template = mcqa_handler_args.copy()
    args_no_template.pop("prompt_template", None) 
    # Ensure prompt_builder_type is default ('mmlu') which expects choices if no template
    args_no_template["prompt_builder_type"] = "mmlu" 


    handler = MultipleChoiceQATaskHandler(
        model=mock_pytorch_model, 
        tokenizer=mock_tokenizer, 
        device="cpu", 
        advanced_args=args_no_template
    )
    handler.logger = MagicMock()

    batch_data_no_choices = {
        "question": ["Q1?"],
        "choices": [[]], # Empty choices for the item
        "label_index": [0],
        "subject": ["Sub1"]
    }
    preds, labels = handler.process_batch(batch_data_no_choices)
    
    assert preds == [] # Expect empty because the item should be skipped
    assert labels == [] # Expect empty because the item should be skipped
    handler.logger.warning.assert_any_call(
        "Skipping item 0 in MCQA due to missing/malformed choices and no custom prompt template (default MMLU prompt needs choices)."
    )

# --- Tests for MultipleChoiceQATaskHandler ---

@pytest.fixture
def mcqa_handler_args(minimal_advanced_config, dummy_task_config_obj):
    # This fixture was defined in the previous response, ensure it's available.
    # It combines minimal_advanced_config with dummy_task_config_obj.handler_options
    # and adds dataset_name and dataset_config_name.
    args = minimal_advanced_config.model_dump(exclude_none=True)
    if dummy_task_config_obj.handler_options:
        args.update(dummy_task_config_obj.handler_options)
    
    # Ensure dataset_name and dataset_config_name are added for the handler's init
    # These would typically be passed by BenchmarkRunner based on the task's dataset config
    if dummy_task_config_obj.datasets:
        args["dataset_name"] = dummy_task_config_obj.datasets[0].name
        args["dataset_config_name"] = dummy_task_config_obj.datasets[0].config
    else: # Fallback if no datasets in dummy_task_config_obj, though a task should have one
        args["dataset_name"] = "dummy_mcqa_dataset"
        args["dataset_config_name"] = "dummy_mcqa_config"
    return args

@patch('src.benchmark.tasks.base_task_handler.setup_logger') # Mocks logger for BaseTaskHandler init
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.PostProcessorFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.MultipleChoiceQATaskHandler._generate_text') # Mock _generate_text
def test_mcqa_handler_process_batch_normal_case(
    mock_generate_text, 
    mock_post_processor_factory, 
    mock_prompt_builder_factory, 
    mock_base_setup_logger, # For BaseTaskHandler's logger
    mock_pytorch_model, # Using existing fixture
    mock_tokenizer,     # Using existing fixture
    mcqa_handler_args   # Using existing fixture
):
    # Mock the logger instance that will be created within BaseTaskHandler
    mock_handler_logger_instance = MagicMock()
    mock_base_setup_logger.return_value = mock_handler_logger_instance

    mock_prompt_builder_instance = MagicMock(name="MockPromptBuilder")
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance
    
    mock_post_processor_instance = MagicMock(name="MockPostProcessor")
    mock_post_processor_factory.get_processor.return_value = mock_post_processor_instance

    handler = MultipleChoiceQATaskHandler(
        model=mock_pytorch_model, 
        tokenizer=mock_tokenizer, 
        device="cpu", 
        advanced_args=mcqa_handler_args
    )
    # Optionally, explicitly set the logger on the handler instance if you need to make assertions on it
    # and want to be sure it's the one you're controlling, though the patch should handle it.
    handler.logger = mock_handler_logger_instance 

    batch_data = {
        "question": ["Q1?", "Q2?"],
        "choices": [["A1","B1","C1","D1"], ["A2","B2","C2","D2"]], # Item-wise choices
        "label_index": [0, 1],
        "subject": ["Sub1", "Sub2"]
    }
    
    expected_prompts = ["Formatted Q1 prompt", "Formatted Q2 prompt"]
    mock_prompt_builder_instance.build_prompts.return_value = expected_prompts
    
    raw_model_outputs_from_generate = ["Raw Model Output 1", "Raw Model Output 2"]
    mock_generate_text.return_value = raw_model_outputs_from_generate
    
    # Expected output from the post-processor
    final_processed_predictions = ["A", "B"] 
    final_processed_labels = ["A", "B"] # Assuming MMLUPostProcessor converts label_index to letter
    mock_post_processor_instance.process.return_value = (final_processed_predictions, final_processed_labels)

    # --- SUT Call ---
    returned_preds, returned_labels = handler.process_batch(batch_data)

    # --- Assertions ---
    assert returned_preds == final_processed_predictions
    assert returned_labels == final_processed_labels

    # Verify PromptBuilder was called correctly
    expected_items_for_prompt_builder = [
        {
            "question": "Q1?", 
            "choices_formatted_str": "A) A1\nB) B1\nC) C1\nD) D1", 
            "subject": "Sub1",
            "choice_A": "A1", "choice_B": "B1", "choice_C": "C1", "choice_D": "D1"
        },
        {
            "question": "Q2?", 
            "choices_formatted_str": "A) A2\nB) B2\nC) C2\nD) D2", 
            "subject": "Sub2",
            "choice_A": "A2", "choice_B": "B2", "choice_C": "C2", "choice_D": "D2"
        }
    ]
    mock_prompt_builder_instance.build_prompts.assert_called_once_with(expected_items_for_prompt_builder)
    
    # Verify _generate_text was called with the prompts from the builder
    mock_generate_text.assert_called_once_with(expected_prompts)
    
    # Verify PostProcessorFactory and the processor instance were called
    # The key comes from handler_options, or defaults to "mmlu_generation"
    expected_postprocessor_key = mcqa_handler_args.get("postprocessor_key", "mmlu_generation")
    mock_post_processor_factory.get_processor.assert_called_once_with(expected_postprocessor_key)
    mock_post_processor_instance.process.assert_called_once_with(
        raw_model_outputs_from_generate, # outputs from _generate_text
        [0, 1],                          # original labels from batch
        batch_data                       # original batch
    )


@patch('src.benchmark.tasks.base_task_handler.setup_logger')
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.MultipleChoiceQATaskHandler._generate_text')
def test_mcqa_handler_process_batch_no_questions_in_batch(
    mock_generate_text,
    mock_prompt_builder_factory,
    mock_base_setup_logger,
    mock_pytorch_model, 
    mock_tokenizer, 
    mcqa_handler_args
):
    mock_handler_logger_instance = MagicMock()
    mock_base_setup_logger.return_value = mock_handler_logger_instance
    mock_prompt_builder_instance = MagicMock()
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance

    handler = MultipleChoiceQATaskHandler(
        model=mock_pytorch_model, tokenizer=mock_tokenizer, device="cpu", advanced_args=mcqa_handler_args
    )
    handler.logger = mock_handler_logger_instance

    batch_data_no_questions = {
        "question": [], # Empty list of questions
        "choices": [],
        "label_index": [],
        "subject": []
    }
    
    predictions, labels = handler.process_batch(batch_data_no_questions)
    
    assert predictions == []
    assert labels == []
    mock_handler_logger_instance.warning.assert_called_with(
        "MCQA batch: Question count (0) and Label count (0) mismatch, or no questions. Skipping batch."
    )
    mock_prompt_builder_instance.build_prompts.assert_not_called()
    mock_generate_text.assert_not_called()


@patch('src.benchmark.tasks.base_task_handler.setup_logger')
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.MultipleChoiceQATaskHandler._generate_text')
def test_mcqa_handler_process_batch_transposed_choices(
    mock_generate_text,
    mock_prompt_builder_factory,
    mock_base_setup_logger,
    mock_pytorch_model,
    mock_tokenizer,
    mcqa_handler_args
):
    mock_handler_logger_instance = MagicMock()
    mock_base_setup_logger.return_value = mock_handler_logger_instance
    mock_prompt_builder_instance = MagicMock()
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance
    mock_generate_text.return_value = ["Raw Output"] # Needs to return a list of same length as processed items

    handler = MultipleChoiceQATaskHandler(
        model=mock_pytorch_model, tokenizer=mock_tokenizer, device="cpu", advanced_args=mcqa_handler_args
    )
    handler.logger = mock_handler_logger_instance

    # Simulate transposed choices: List of choice types, where each inner list is for the batch
    # E.g., choices[0] = all 'A' choices, choices[1] = all 'B' choices
    batch_data_transposed_choices = {
        "question": ["Q1?"],
        "choices": [["A1_text"], ["B1_text"], ["C1_text"], ["D1_text"]], # Transposed structure
        "label_index": [0],
        "subject": ["Sub1"]
    }

    expected_prompts = ["Formatted Q1 with transposed choices"]
    mock_prompt_builder_instance.build_prompts.return_value = expected_prompts
    
    # Mock post-processor to avoid its complexity for this specific test unit
    with patch.object(handler, '_post_process', return_value=(["Pred A"], ["Label A"])) as mock_post_proc:
        preds, labels = handler.process_batch(batch_data_transposed_choices)

    assert preds == ["Pred A"]
    assert labels == ["Label A"]
    
    handler.logger.debug.assert_any_call("MCQA choices format: transposed (num_options x batch_size). Num_options: 4. Restructuring.")
    
    expected_item_for_builder = [{
        "question": "Q1?",
        "choices_formatted_str": "A) A1_text\nB) B1_text\nC) C1_text\nD) D1_text",
        "subject": "Sub1",
        "choice_A": "A1_text", "choice_B": "B1_text", "choice_C": "C1_text", "choice_D": "D1_text"
    }]
    mock_prompt_builder_instance.build_prompts.assert_called_once_with(expected_item_for_builder)
    mock_generate_text.assert_called_once_with(expected_prompts)
    mock_post_proc.assert_called_once()


@patch('src.benchmark.tasks.base_task_handler.setup_logger')
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.MultipleChoiceQATaskHandler._generate_text')
def test_mcqa_handler_process_batch_missing_choices_no_template_skips_item(
    mock_generate_text,
    mock_prompt_builder_factory,
    mock_base_setup_logger,
    mock_pytorch_model,
    mock_tokenizer,
    mcqa_handler_args # This fixture uses a default "mmlu" prompt_builder_type and has a template
):
    mock_handler_logger_instance = MagicMock()
    mock_base_setup_logger.return_value = mock_handler_logger_instance
    mock_prompt_builder_instance = MagicMock()
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance

    # Create a copy of args and remove the prompt_template to trigger the specific path
    args_for_this_test = mcqa_handler_args.copy()
    args_for_this_test.pop("prompt_template", None)
    # Ensure prompt_builder_type remains "mmlu" or similar that would by default expect choices
    args_for_this_test["prompt_builder_type"] = "mmlu" 

    handler = MultipleChoiceQATaskHandler(
        model=mock_pytorch_model, tokenizer=mock_tokenizer, device="cpu", advanced_args=args_for_this_test
    )
    handler.logger = mock_handler_logger_instance

    batch_data_missing_choices = {
        "question": ["Q1?"],
        "choices": [[]], # Empty list for choices for this item
        "label_index": [0],
        "subject": ["Sub1"]
    }
    
    # Mock post-processor because it won't be reached if items are skipped
    with patch.object(handler, '_post_process') as mock_post_proc:
        preds, labels = handler.process_batch(batch_data_missing_choices)

    assert preds == [] # Item should be skipped
    assert labels == [] # Item should be skipped
    mock_handler_logger_instance.warning.assert_any_call(
        "Skipping item 0 in MCQA due to missing/malformed choices and no custom prompt template (default MMLU prompt needs choices)."
    )
    mock_prompt_builder_instance.build_prompts.assert_not_called() # or called with an empty list
    mock_generate_text.assert_not_called()
    mock_post_proc.assert_not_called()


# --- Fixture for SummarizationTaskHandler ---
@pytest.fixture
def summarization_handler_args(minimal_advanced_config, dummy_task_config_obj):
    args = minimal_advanced_config.model_dump(exclude_none=True)
    # Summarization might have specific handler_options in a real TaskConfig
    # For this test, we'll assume it might use a default prompt or one from args.
    # Ensure 'prompt_template' could be present or absent.
    if dummy_task_config_obj.handler_options: # Or a specific task_config for summarization
        args.update(dummy_task_config_obj.handler_options)
    # Add dataset specific info that _run_task_evaluation would pass if needed by the handler's init
    # args["dataset_name"] = "cnn_dailymail" # Example
    # args["dataset_config_name"] = "3.0.0" # Example
    return args

# --- Tests for SummarizationTaskHandler ---

@patch('src.benchmark.tasks.base_task_handler.setup_logger') # Mocks logger for BaseTaskHandler init
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.PostProcessorFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.SummarizationTaskHandler._generate_text')
def test_summarization_handler_process_batch(
    mock_generate_text,
    mock_post_processor_factory,
    mock_prompt_builder_factory,
    mock_base_setup_logger,
    mock_pytorch_model,
    mock_tokenizer,
    summarization_handler_args # Use the specific fixture
):
    mock_handler_logger_instance = MagicMock()
    mock_base_setup_logger.return_value = mock_handler_logger_instance

    mock_prompt_builder_instance = MagicMock(name="MockPromptBuilder")
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance
    
    mock_post_processor_instance = MagicMock(name="MockPostProcessor")
    mock_post_processor_factory.get_processor.return_value = mock_post_processor_instance

    handler = SummarizationTaskHandler(
        model=mock_pytorch_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        advanced_args=summarization_handler_args
    )
    handler.logger = mock_handler_logger_instance

    batch_data = {
        "input_text": ["This is a long article to summarize.", "Another news piece requiring a summary."],
        "target_text": ["Short summary 1.", "Concise summary 2."] # Reference summaries
    }
    
    expected_prompts = ["Prompt for article 1", "Prompt for article 2"]
    mock_prompt_builder_instance.build_prompts.return_value = expected_prompts
    
    raw_model_outputs = ["Generated summary 1.", "Generated summary 2."]
    mock_generate_text.return_value = raw_model_outputs
    
    final_processed_predictions = ["Cleaned summary 1", "Cleaned summary 2"]
    final_processed_labels = ["Short summary 1.", "Concise summary 2."] # Assuming postproc passes labels through
    mock_post_processor_instance.process.return_value = (final_processed_predictions, final_processed_labels)

    # --- SUT Call ---
    returned_preds, returned_labels = handler.process_batch(batch_data)

    # --- Assertions ---
    assert returned_preds == final_processed_predictions
    assert returned_labels == final_processed_labels

    expected_items_for_builder = [
        {"input_text": "This is a long article to summarize.", "article_text": "This is a long article to summarize."},
        {"input_text": "Another news piece requiring a summary.", "article_text": "Another news piece requiring a summary."}
    ]
    mock_prompt_builder_instance.build_prompts.assert_called_once_with(expected_items_for_builder)
    mock_generate_text.assert_called_once_with(expected_prompts)
    
    expected_postprocessor_key = summarization_handler_args.get("postprocessor_key", "summarization")
    mock_post_processor_factory.get_processor.assert_called_once_with(expected_postprocessor_key)
    mock_post_processor_instance.process.assert_called_once_with(
        raw_model_outputs,
        batch_data["target_text"],
        batch_data
    )

# --- Fixture for TranslationTaskHandler ---
@pytest.fixture
def translation_handler_args(minimal_advanced_config, dummy_task_config_obj):
    args = minimal_advanced_config.model_dump(exclude_none=True)
    if dummy_task_config_obj.handler_options: # Or a specific task_config for translation
        args.update(dummy_task_config_obj.handler_options)
    # Translation handler requires dataset_config_name for lang parsing
    args["dataset_config_name"] = "en-fr" # Example
    args["dataset_name"] = "opus100" # Example
    return args

# --- Tests for TranslationTaskHandler ---
@patch('src.benchmark.tasks.base_task_handler.setup_logger')
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.PostProcessorFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.TranslationTaskHandler._generate_text')
def test_translation_handler_process_batch(
    mock_generate_text,
    mock_post_processor_factory,
    mock_prompt_builder_factory,
    mock_base_setup_logger,
    mock_pytorch_model,
    mock_tokenizer,
    translation_handler_args 
):
    mock_handler_logger_instance = MagicMock()
    mock_base_setup_logger.return_value = mock_handler_logger_instance

    mock_prompt_builder_instance = MagicMock(name="MockPromptBuilder")
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance
    
    mock_post_processor_instance = MagicMock(name="MockPostProcessor")
    mock_post_processor_factory.get_processor.return_value = mock_post_processor_instance

    handler = TranslationTaskHandler(
        model=mock_pytorch_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        advanced_args=translation_handler_args
    )
    handler.logger = mock_handler_logger_instance

    batch_data = {
        "input_text": ["Hello world.", "Good morning."],
        "target_text": ["Bonjour le monde.", "Bonjour."] 
    }
    
    expected_prompts = ["Translate: Hello world.", "Translate: Good morning."]
    mock_prompt_builder_instance.build_prompts.return_value = expected_prompts
    
    raw_model_outputs = ["Bonjour le monde output.", "Bonjour output."]
    mock_generate_text.return_value = raw_model_outputs
    
    final_processed_predictions = ["Bonjour le monde output", "Bonjour output"] # Assuming simple post-processing
    final_processed_labels = ["Bonjour le monde.", "Bonjour."]
    mock_post_processor_instance.process.return_value = (final_processed_predictions, final_processed_labels)

    returned_preds, returned_labels = handler.process_batch(batch_data)

    assert returned_preds == final_processed_predictions
    assert returned_labels == final_processed_labels

    expected_items_for_builder = [
        {"input_text": "Hello world."},
        {"input_text": "Good morning."}
    ]
    mock_prompt_builder_instance.build_prompts.assert_called_once_with(expected_items_for_builder)
    mock_generate_text.assert_called_once_with(expected_prompts)
    
    expected_postprocessor_key = translation_handler_args.get("postprocessor_key", "translation")
    mock_post_processor_factory.get_processor.assert_called_once_with(expected_postprocessor_key)
    mock_post_processor_instance.process.assert_called_once_with(
        raw_model_outputs,
        batch_data["target_text"],
        batch_data
    )

# --- Fixture for MathReasoningGenerationTaskHandler ---
@pytest.fixture
def math_handler_args(minimal_advanced_config, dummy_task_config_obj):
    args = minimal_advanced_config.model_dump(exclude_none=True)
    # Math task might have specific options like a postprocessor_key for GSM8K
    if dummy_task_config_obj.handler_options: # Or a specific task_config for math
        args.update(dummy_task_config_obj.handler_options)
    args["postprocessor_key"] = args.get("postprocessor_key", "gsm8k") # Ensure it's set
    return args

# --- Tests for MathReasoningGenerationTaskHandler ---
@patch('src.benchmark.tasks.base_task_handler.setup_logger')
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.PostProcessorFactory') # For the _post_process_for_exact_match
@patch('src.benchmark.tasks.concrete_task_handlers.MathReasoningGenerationTaskHandler._generate_text')
def test_math_reasoning_handler_process_batch(
    mock_generate_text,
    mock_post_processor_factory, # This mock is for the one used in _post_process_for_exact_match
    mock_prompt_builder_factory,
    mock_base_setup_logger,
    mock_pytorch_model,
    mock_tokenizer,
    math_handler_args
):
    mock_handler_logger_instance = MagicMock()
    mock_base_setup_logger.return_value = mock_handler_logger_instance

    mock_prompt_builder_instance = MagicMock(name="MockPromptBuilder")
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance
    
    mock_exact_match_post_processor_instance = MagicMock(name="MockGSM8KPostProcessor")
    # This factory is called inside _post_process_for_exact_match
    mock_post_processor_factory.get_processor.return_value = mock_exact_match_post_processor_instance


    handler = MathReasoningGenerationTaskHandler(
        model=mock_pytorch_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        advanced_args=math_handler_args
    )
    handler.logger = mock_handler_logger_instance

    batch_data = {
        "input_text": ["What is 2+2?", "Solve for x: x*3=9"],
        "target_text": ["The answer is ####4", "The solution is ####3"] 
    }
    
    expected_prompts = ["Solve: What is 2+2?", "Solve: Solve for x: x*3=9"]
    mock_prompt_builder_instance.build_prompts.return_value = expected_prompts
    
    raw_model_outputs = ["2+2 = 4. The answer is ####4", "x = 9/3 so x = 3. The solution is ####3"]
    mock_generate_text.return_value = raw_model_outputs
    
    # Output from the _post_process_for_exact_match
    exact_match_preds = ["4", "3"]
    exact_match_labels = ["4", "3"]
    mock_exact_match_post_processor_instance.process.return_value = (exact_match_preds, exact_match_labels)

    # --- SUT Call ---
    result_dict = handler.process_batch(batch_data)

    # --- Assertions ---
    assert result_dict["text_predictions"] == raw_model_outputs
    assert result_dict["exact_match_predictions"] == exact_match_preds
    assert result_dict["labels_for_text"] == batch_data["target_text"]
    assert result_dict["labels_for_exact_match"] == exact_match_labels

    expected_items_for_builder = [
        {"input_text": "What is 2+2?"},
        {"input_text": "Solve for x: x*3=9"}
    ]
    mock_prompt_builder_instance.build_prompts.assert_called_once_with(expected_items_for_builder)
    mock_generate_text.assert_called_once_with(expected_prompts)
    
    # Assert PostProcessorFactory was called for the gsm8k (or configured) key
    expected_postprocessor_key = math_handler_args.get("postprocessor_key", "gsm8k")
    mock_post_processor_factory.get_processor.assert_called_once_with(expected_postprocessor_key)
    mock_exact_match_post_processor_instance.process.assert_called_once_with(
        raw_model_outputs,
        batch_data["target_text"],
        batch_data
    )

# --- Tests for GlueClassificationPromptingTaskHandler ---

@patch('src.benchmark.tasks.base_task_handler.setup_logger')
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.PostProcessorFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.GlueClassificationPromptingTaskHandler._generate_text')
def test_glue_sst2_handler_process_batch(
    mock_generate_text,
    mock_post_processor_factory,
    mock_prompt_builder_factory,
    mock_base_setup_logger,
    mock_pytorch_model,
    mock_tokenizer,
    glue_sst2_handler_args
):
    mock_handler_logger_instance = MagicMock()
    mock_base_setup_logger.return_value = mock_handler_logger_instance
    mock_prompt_builder_instance = MagicMock(name="MockGlueSST2PromptBuilder")
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance
    mock_post_processor_instance = MagicMock(name="MockGlueSST2PostProcessor")
    mock_post_processor_factory.get_processor.return_value = mock_post_processor_instance

    handler = GlueClassificationPromptingTaskHandler(
        model=mock_pytorch_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        advanced_args=glue_sst2_handler_args
    )
    handler.logger = mock_handler_logger_instance

    batch_data = {
        "input_text": ["This movie is great.", "This movie is terrible."],
        "label_index": [1, 0] # SST-2: 1 for positive, 0 for negative
    }
    
    expected_prompts = ["SST-2 Prompt: This movie is great.", "SST-2 Prompt: This movie is terrible."]
    mock_prompt_builder_instance.build_prompts.return_value = expected_prompts
    
    raw_model_outputs = ["positive output", "negative output"]
    mock_generate_text.return_value = raw_model_outputs
    
    final_processed_predictions = [1, 0] # Post-processor maps to 0 or 1
    final_processed_labels = [1, 0]
    mock_post_processor_instance.process.return_value = (final_processed_predictions, final_processed_labels)

    returned_preds, returned_labels = handler.process_batch(batch_data)

    assert returned_preds == final_processed_predictions
    assert returned_labels == final_processed_labels

    expected_items_for_builder = [
        {"input_text": "This movie is great.", "dataset_config_name": "sst2"},
        {"input_text": "This movie is terrible.", "dataset_config_name": "sst2"}
    ]
    mock_prompt_builder_instance.build_prompts.assert_called_once_with(expected_items_for_builder)
    mock_generate_text.assert_called_once_with(expected_prompts)
    
    mock_post_processor_factory.get_processor.assert_called_once_with(glue_sst2_handler_args["postprocessor_key"])
    mock_post_processor_instance.process.assert_called_once_with(
        raw_model_outputs, batch_data["label_index"], batch_data
    )

# --- Tests for GlueTextPairPromptingTaskHandler ---

@patch('src.benchmark.tasks.base_task_handler.setup_logger')
@patch('src.benchmark.tasks.concrete_task_handlers.PromptBuilderFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.PostProcessorFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.GlueTextPairPromptingTaskHandler._generate_text')
def test_glue_mrpc_handler_process_batch(
    mock_generate_text,
    mock_post_processor_factory,
    mock_prompt_builder_factory,
    mock_base_setup_logger,
    mock_pytorch_model,
    mock_tokenizer,
    glue_mrpc_handler_args 
):
    mock_handler_logger_instance = MagicMock()
    mock_base_setup_logger.return_value = mock_handler_logger_instance
    mock_prompt_builder_instance = MagicMock(name="MockGlueMRPCPromptBuilder")
    mock_prompt_builder_factory.get_builder.return_value = mock_prompt_builder_instance
    mock_post_processor_instance = MagicMock(name="MockGlueMRPCPostProcessor")
    mock_post_processor_factory.get_processor.return_value = mock_post_processor_instance

    handler = GlueTextPairPromptingTaskHandler(
        model=mock_pytorch_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        advanced_args=glue_mrpc_handler_args
    )
    handler.logger = mock_handler_logger_instance

    batch_data = {
        "input_text_pair1": ["Sentence 1a", "Sentence 2a"],
        "input_text_pair2": ["Sentence 1b", "Sentence 2b"],
        "label_index": [1, 0] # MRPC: 1 for paraphrase, 0 for not
    }
    
    expected_prompts = ["MRPC Prompt 1", "MRPC Prompt 2"]
    mock_prompt_builder_instance.build_prompts.return_value = expected_prompts
    
    raw_model_outputs = ["yes, they are paraphrases", "no, not paraphrases"]
    mock_generate_text.return_value = raw_model_outputs
    
    final_processed_predictions = [1, 0] # Post-processor maps to 0 or 1
    final_processed_labels = [1, 0]
    mock_post_processor_instance.process.return_value = (final_processed_predictions, final_processed_labels)

    returned_preds, returned_labels = handler.process_batch(batch_data)

    assert returned_preds == final_processed_predictions
    assert returned_labels == final_processed_labels

    expected_items_for_builder = [
        {"input_text_pair1": "Sentence 1a", "input_text_pair2": "Sentence 1b", "dataset_config_name": "mrpc"},
        {"input_text_pair1": "Sentence 2a", "input_text_pair2": "Sentence 2b", "dataset_config_name": "mrpc"}
    ]
    mock_prompt_builder_instance.build_prompts.assert_called_once_with(expected_items_for_builder)
    mock_generate_text.assert_called_once_with(expected_prompts)
    
    mock_post_processor_factory.get_processor.assert_called_once_with(glue_mrpc_handler_args["postprocessor_key"])
    mock_post_processor_instance.process.assert_called_once_with(
        raw_model_outputs, batch_data["label_index"], batch_data
    )

# --- Tests for CustomScriptTaskHandler ---

@patch('src.benchmark.tasks.base_task_handler.setup_logger') # For BaseTaskHandler
@patch('src.benchmark.tasks.concrete_task_handlers.importlib.util') # To mock script loading
@patch('src.benchmark.tasks.base_task_handler.PostProcessorFactory')
@patch('src.benchmark.tasks.concrete_task_handlers.CustomScriptTaskHandler._generate_text')
def test_custom_script_handler_process_batch_success(
    mock_generate_text,
    mock_post_processor_factory,
    mock_importlib_util,
    mock_base_setup_logger,
    mock_pytorch_model,
    mock_tokenizer,
    custom_script_handler_args # Uses the fixture that creates a dummy script
):
    mock_handler_logger_instance = MagicMock()
    mock_base_setup_logger.return_value = mock_handler_logger_instance

    # Mock the loaded custom function
    mock_custom_batch_proc_fn = MagicMock(name="my_custom_batch_processor_mock")
    # Setup the module loading mock
    mock_spec = MagicMock()
    mock_module = MagicMock()
    setattr(mock_module, custom_script_handler_args["handler_function_name"], mock_custom_batch_proc_fn)
    mock_importlib_util.spec_from_file_location.return_value = mock_spec
    mock_importlib_util.module_from_spec.return_value = mock_module
    # mock_spec.loader.exec_module is implicitly handled if module_from_spec returns the module with the attr

    mock_post_processor_instance = MagicMock(name="MockPostProcessor")
    mock_post_processor_factory.get_processor.return_value = mock_post_processor_instance

    handler = CustomScriptTaskHandler(
        model=mock_pytorch_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        advanced_args=custom_script_handler_args
    )
    handler.logger = mock_handler_logger_instance

    batch_data = {
        "input_text": ["item1", "item2"],
        "label_index": ["labelA", "labelB"]
    }
    
    # Output from the mocked custom_batch_proc_fn
    prompts_from_custom_fn = ["Custom prompt for: item1", "Custom prompt for: item2"]
    labels_from_custom_fn = ["labelA", "labelB"]
    mock_custom_batch_proc_fn.return_value = (prompts_from_custom_fn, labels_from_custom_fn)
    
    raw_model_outputs_from_generate = ["Model output for item1", "Model output for item2"]
    mock_generate_text.return_value = raw_model_outputs_from_generate
    
    final_processed_predictions = ["Processed output1", "Processed output2"]
    final_processed_labels = ["Processed labelA", "Processed labelB"]
    mock_post_processor_instance.process.return_value = (final_processed_predictions, final_processed_labels)

    # --- SUT Call ---
    returned_preds, returned_labels = handler.process_batch(batch_data)

    # --- Assertions ---
    assert returned_preds == final_processed_predictions
    assert returned_labels == final_processed_labels

    mock_custom_batch_proc_fn.assert_called_once_with(
        batch=batch_data,
        model=mock_pytorch_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        advanced_args=custom_script_handler_args, # The handler passes its own advanced_args
        script_args=custom_script_handler_args["handler_script_args"]
    )
    mock_generate_text.assert_called_once_with(prompts_from_custom_fn)
    
    expected_postprocessor_key = custom_script_handler_args.get("postprocessor_key", "generic_generation")
    # Prepare expected options for PostProcessorFactory, including script-related ones if any
    postproc_options = {
        "script_path": custom_script_handler_args.get("postprocessor_script_path"),
        "function_name": custom_script_handler_args.get("postprocessor_function_name"),
        "script_args": custom_script_handler_args.get("postprocessor_script_args", {}),
    }
    if "postprocessor_options" in custom_script_handler_args: # Merge if general options are also present
        postproc_options.update(custom_script_handler_args["postprocessor_options"])
    
    # Filter out None script_path/function_name before passing to factory mock
    final_postproc_options = {k: v for k, v in postproc_options.items() 
                                if not (k in ["script_path", "function_name"] and v is None)}


    mock_post_processor_factory.get_processor.assert_called_once_with(expected_postprocessor_key, processor_options=final_postproc_options)
    mock_post_processor_instance.process.assert_called_once_with(
        raw_model_outputs_from_generate, labels_from_custom_fn, batch_data
    )


@patch('src.benchmark.tasks.base_task_handler.setup_logger')
def test_custom_script_handler_init_missing_script_path(mock_base_setup_logger, minimal_advanced_config, mock_pytorch_model, mock_tokenizer):
    mock_base_setup_logger.return_value = MagicMock()
    args = minimal_advanced_config.model_dump(exclude_none=True)
    args["handler_function_name"] = "some_function"
    # "handler_script_path" is missing
    
    with pytest.raises(ValueError, match="CustomScriptTaskHandler requires 'handler_script_path' and 'handler_function_name' in handler_options."):
        CustomScriptTaskHandler(
            model=mock_pytorch_model, tokenizer=mock_tokenizer, device="cpu", advanced_args=args
        )

@patch('src.benchmark.tasks.base_task_handler.setup_logger')
def test_custom_script_handler_init_script_not_found(mock_base_setup_logger, minimal_advanced_config, mock_pytorch_model, mock_tokenizer, tmp_path):
    mock_base_setup_logger.return_value = MagicMock()
    args = minimal_advanced_config.model_dump(exclude_none=True)
    args["handler_script_path"] = str(tmp_path / "non_existent_script.py")
    args["handler_function_name"] = "some_function"
    
    with pytest.raises(FileNotFoundError, match="Handler script not found"):
        CustomScriptTaskHandler(
            model=mock_pytorch_model, tokenizer=mock_tokenizer, device="cpu", advanced_args=args
        )

@patch('src.benchmark.tasks.base_task_handler.setup_logger')
@patch('src.benchmark.tasks.concrete_task_handlers.importlib.util')
def test_custom_script_handler_init_function_not_in_script(
    mock_importlib_util, mock_base_setup_logger, minimal_advanced_config, 
    mock_pytorch_model, mock_tokenizer, tmp_path
):
    mock_base_setup_logger.return_value = MagicMock()
    
    dummy_script_content = "def another_function(): pass"
    script_file = tmp_path / "dummy_script_no_func.py"
    script_file.write_text(dummy_script_content)

    mock_spec = MagicMock()
    mock_module = MagicMock()
    # Simulate the module does NOT have the target function
    if hasattr(mock_module, "my_target_function"): # Should not exist
        delattr(mock_module, "my_target_function")

    mock_importlib_util.spec_from_file_location.return_value = mock_spec
    mock_importlib_util.module_from_spec.return_value = mock_module
    # exec_module is called on spec.loader, not importlib.util directly
    # mock_spec.loader.exec_module = MagicMock() # Ensure it can be called

    args = minimal_advanced_config.model_dump(exclude_none=True)
    args["handler_script_path"] = str(script_file)
    args["handler_function_name"] = "my_target_function" # This function is not in dummy_script_content

    with pytest.raises(AttributeError, match="Function 'my_target_function' not found in script"):
        CustomScriptTaskHandler(
            model=mock_pytorch_model, tokenizer=mock_tokenizer, device="cpu", advanced_args=args
        )