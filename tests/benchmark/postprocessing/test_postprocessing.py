import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.benchmark.postprocessing.concrete_postprocessors import (
    DefaultPostProcessor,
    MMLUPostProcessor,
    GSM8KPostProcessor,
    SummarizationPostProcessor,
    TranslationPostProcessor,
    GlueSST2OutputPostProcessor,
    GlueMRPCOutputPostProcessor,
    GlueSTSBOutputPostProcessor,
    CreativeTextPostProcessor,
    CustomScriptPostProcessor
)

# --- Test DefaultPostProcessor ---
def test_default_postprocessor():
    processor = DefaultPostProcessor()
    preds = ["Raw prediction 1", "Another raw one"]
    labels = [123, {"key": "value"}]
    processed_preds, processed_labels = processor.process(preds, labels)

    assert processed_preds == preds # Should remain unchanged
    assert processed_labels == ["123", "{'key': 'value'}"] # Labels converted to string

def test_default_postprocessor_empty():
    processor = DefaultPostProcessor()
    processed_preds, processed_labels = processor.process([], [])
    assert processed_preds == []
    assert processed_labels == []

# --- Test MMLUPostProcessor ---
def test_mmlu_postprocessor():
    processor = MMLUPostProcessor()
    preds = ["  (A) Correct", "b is the answer", "Option C.", "D", "Invalid E then A"]
    labels = [0, 1, 2, 3, 0] # Corresponding to A, B, C, D, A
    
    processed_preds, processed_labels = processor.process(preds, labels)
    
    assert processed_preds == ["A", "B", "C", "D", "A"]
    assert processed_labels == ["A", "B", "C", "D", "A"]

def test_mmlu_postprocessor_invalid_inputs():
    processor = MMLUPostProcessor()
    # Changed the first prediction to not contain A, B, C, or D
    preds = ["No valid options xyz", ""] 
    labels = [5, "invalid_label"]
    processed_preds, processed_labels = processor.process(preds, labels)
    assert processed_preds == ["INVALID_PRED", "INVALID_PRED"]
    assert processed_labels == ["INVALID_LABEL", "INVALID_LABEL"]

# --- Test GSM8KPostProcessor ---
def test_gsm8k_postprocessor():
    processor = GSM8KPostProcessor()
    preds = [
        "The final answer is ####42.", 
        "So the result is #### 1,234", 
        "####-5.5", 
        "No answer here"
    ]
    labels = [
        "The solution is ####42", 
        " ####1234", 
        "The number should be ####-5.5",
        "No number #### here"
    ]
    processed_preds, processed_labels = processor.process(preds, labels)
    
    assert processed_preds == ["42", "1234", "-5.5", "ANSWER_NOT_FOUND"]
    assert processed_labels == ["42", "1234", "-5.5", "ANSWER_NOT_FOUND"]

# --- Test SummarizationPostProcessor ---
def test_summarization_postprocessor():
    processor = SummarizationPostProcessor() # Inherits from Default
    preds = ["This is a summary.", "  Another one.  "]
    labels = ["Reference summary 1.", "Ref 2."]
    processed_preds, processed_labels = processor.process(preds, labels)
    # DefaultPostProcessor only stringifies labels, does not change preds
    assert processed_preds == ["This is a summary.", "  Another one.  "]
    assert processed_labels == ["Reference summary 1.", "Ref 2."]


# --- Test TranslationPostProcessor ---
def test_translation_postprocessor():
    processor = TranslationPostProcessor() # Inherits from Default
    preds = ["Bonjour le monde.", "  Hola.  "]
    labels = ["Hello world.", "Hi."]
    processed_preds, processed_labels = processor.process(preds, labels)
    assert processed_preds == ["Bonjour le monde.", "  Hola.  "]
    assert processed_labels == ["Hello world.", "Hi."]

# --- Test GlueSST2OutputPostProcessor ---
def test_glue_sst2_output_postprocessor():
    processor = GlueSST2OutputPostProcessor()
    preds = ["This is positive.", "  negative output.", "neutral or POS", " neg ", "unknown"]
    labels = [1, 0, 1, 0, 1] # positive, negative, positive, negative, positive
    
    processed_preds, processed_labels = processor.process(preds, labels)
    
    assert processed_preds == [1, 0, 1, 0, 0] # Last one defaults to 0 (invalid)
    assert processed_labels == [1, 0, 1, 0, 1]

# --- Test GlueMRPCOutputPostProcessor ---
def test_glue_mrpc_output_postprocessor():
    processor = GlueMRPCOutputPostProcessor()
    preds = ["yes, they are paraphrases", "no", "  equivalent ", "not equivalent.", "maybe"]
    labels = [1, 0, 1, 0, 0] # equivalent, not, equivalent, not, not
    
    processed_preds, processed_labels = processor.process(preds, labels)
    
    assert processed_preds == [1, 0, 1, 0, 0] # Last one defaults to 0
    assert processed_labels == [1, 0, 1, 0, 0]

# --- Test GlueSTSBOutputPostProcessor ---
def test_glue_stsb_output_postprocessor():
    processor = GlueSTSBOutputPostProcessor()
    preds = ["Similarity Score: 4.5", "Output is 3.0", "around 2", "1", "high similarity (5.0)", "text 0.0 text"]
    labels = [4.5, 3.0, 2.0, 1.0, 5.0, 0.0]
    
    processed_preds, processed_labels = processor.process(preds, labels)
    
    assert processed_preds == [4.5, 3.0, 2.0, 1.0, 5.0, 0.0]
    assert processed_labels == [4.5, 3.0, 2.0, 1.0, 5.0, 0.0]

def test_glue_stsb_output_postprocessor_invalid():
    processor = GlueSTSBOutputPostProcessor()
    preds = ["no score here", "Score: ten"]
    labels = [1.0, "invalid_label_type"]
    processed_preds, processed_labels = processor.process(preds, labels)
    assert processed_preds == [-1.0, -1.0] # Default invalid score
    assert processed_labels == [1.0, -1.0] # Default invalid score for label


# --- Test CreativeTextPostProcessor ---
def test_creative_text_postprocessor():
    processor = CreativeTextPostProcessor()
    preds = [
        "  This is a \n creative \n\n piece.  Lot's  of \t spaces.\n\n\n  Next paragraph.  ", # Added more varied spacing
        None,
        "Single sentence."
    ]
    labels = ["Reference text 1", None, "Ref 3"]

    processed_preds_list, processed_labels = processor.process(preds, labels)

    # Expected after new cleaning logic:
    # 1. strip: "This is a \n creative \n\n piece.  Lot's  of \t spaces.\n\n\n  Next paragraph."
    # 2. normalize line breaks (assume already \n)
    # 3. \n\s*\n+ to %%%PARA_BREAK%%%: 
    #    "This is a \n creative %%%PARA_BREAK%%% piece.  Lot's  of \t spaces. %%%PARA_BREAK%%%  Next paragraph."
    # 4. \n to space: "This is a   creative %%%PARA_BREAK%%% piece.  Lot's  of \t spaces. %%%PARA_BREAK%%%  Next paragraph."
    # 5. " ".join(split()): "This is a creative %%%PARA_BREAK%%% piece. Lot's of spaces. %%%PARA_BREAK%%% Next paragraph."
    # 6. replace %%%PARA_BREAK%%%: "This is a creative\n\npiece. Lot's of spaces.\n\nNext paragraph."
    expected_cleaned_1 = "This is a creative\n\npiece. Lot's of spaces.\n\nNext paragraph."

    assert processed_preds_list[0]["cleaned_text"] == expected_cleaned_1
    assert processed_preds_list[0]["word_count"] == 10 # "This is a creative piece. Lot's of spaces. Next paragraph." -> 10 words
    assert processed_preds_list[0]["original_text"] == preds[0]

    assert processed_preds_list[1]["cleaned_text"] == ""
    assert processed_preds_list[1]["word_count"] == 0
    assert processed_preds_list[1]["original_text"] is None

    assert processed_preds_list[2]["cleaned_text"] == "Single sentence."
    assert processed_preds_list[2]["word_count"] == 2

    assert processed_labels == ["Reference text 1", None, "Ref 3"]


# --- Test CustomScriptPostProcessor ---
@pytest.fixture
def dummy_postprocessor_script(tmp_path: Path) -> Path:
    script_content = """
from typing import List, Dict, Any, Tuple, Optional

def my_custom_postproc(
    predictions: List[Any], 
    labels: List[Any], 
    batch: Optional[Dict[str, Any]] = None,
    script_args: Optional[Dict[str, Any]] = None
) -> Tuple[List[Any], List[Any]]:
    processed_preds = [f"processed_{p}" for p in predictions]
    processed_labels = [f"processed_{l}" for l in labels]
    if script_args and script_args.get("add_batch_info") and batch:
        processed_preds = [f"{pp} (batch_id:{batch.get('id')})" for pp in processed_preds]
    return processed_preds, processed_labels
"""
    script_file = tmp_path / "custom_postproc_functions.py"
    script_file.write_text(script_content)
    return script_file

def test_custom_script_postprocessor_success(dummy_postprocessor_script: Path):
    processor = CustomScriptPostProcessor(
        script_path=str(dummy_postprocessor_script),
        function_name="my_custom_postproc",
        script_args={"add_batch_info": True}
    )
    preds = ["pred1", "pred2"]
    labels = ["label1", "label2"]
    batch_data = {"id": "batch001"}
    
    processed_preds, processed_labels = processor.process(preds, labels, batch_data)
    
    assert processed_preds == ["processed_pred1 (batch_id:batch001)", "processed_pred2 (batch_id:batch001)"]
    assert processed_labels == ["processed_label1", "processed_label2"]

def test_custom_script_postprocessor_no_script_acts_as_default():
    processor = CustomScriptPostProcessor() # No script_path or function_name
    preds = ["pred1"]
    labels = [123]
    processed_preds, processed_labels = processor.process(preds, labels)
    assert processed_preds == ["pred1"]
    assert processed_labels == ["123"] # Default behavior

def test_custom_script_postprocessor_script_error(dummy_postprocessor_script: Path):
    # Test when the custom script itself raises an error
    error_script_content = """
def my_error_postproc(predictions, labels, batch, script_args):
    raise ValueError("Intentional script error")
"""
    error_script_file = dummy_postprocessor_script.parent / "error_postproc_script.py"
    error_script_file.write_text(error_script_content)

    processor = CustomScriptPostProcessor(
        script_path=str(error_script_file),
        function_name="my_error_postproc"
    )
    preds = ["p1"]
    labels = ["l1"]
    # Expect it to fall back to DefaultPostProcessor's behavior
    processed_preds, processed_labels = processor.process(preds, labels)
    assert processed_preds == ["p1"] # Unchanged by default
    assert processed_labels == ["l1"] # Stringified by default

def test_custom_script_postprocessor_init_errors():
    with pytest.raises(ValueError, match="CustomScriptPostProcessor requires both 'script_path' and 'function_name'"):
        CustomScriptPostProcessor(script_path="somepath.py")
    
    with pytest.raises(FileNotFoundError):
        CustomScriptPostProcessor(script_path="nonexistent.py", function_name="func")

@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
def test_custom_script_postprocessor_function_not_in_script(mock_module_from_spec, mock_spec_from_file_location, tmp_path):
    script_content = "def another_func(): pass"
    p = tmp_path / "script_no_func.py"
    p.write_text(script_content)

    mock_spec = MagicMock()
    mock_spec.loader = MagicMock()
    mock_spec.loader.exec_module = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec
    
    mock_custom_module = MagicMock()
    # Ensure the module does NOT have 'target_func'
    if hasattr(mock_custom_module, "target_func"):
        delattr(mock_custom_module, "target_func")
    mock_module_from_spec.return_value = mock_custom_module
    
    with pytest.raises(AttributeError, match="Function 'target_func' not found in script"):
        CustomScriptPostProcessor(script_path=str(p), function_name="target_func")