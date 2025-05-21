from typing import Any
import pytest
from unittest.mock import patch, MagicMock # Added MagicMock

from src.benchmark.prompting.base_prompt_builder import BasePromptBuilder
from src.benchmark.prompting.concrete_prompt_builders import (
    TemplateBasedPromptBuilder,
    MMLUPromptBuilder,
    TranslationPromptBuilder,
    GlueSST2PromptBuilder,
    GlueMRPCPromptBuilder,
    GlueSTSBPromptBuilder
)
from src.benchmark.prompting.prompt_builder_factory import PromptBuilderFactory

# --- Tests for BasePromptBuilder (indirectly via a simple concrete implementation) ---

class ConcreteTestPromptBuilder(BasePromptBuilder): # For testing abstract methods if needed
    def build_prompts(self, batch_items):
        if not self.template_string:
            # Fallback if no template: try to get 'text', then 'input_text', then 'question'
            prompts = []
            for item in batch_items:
                text_content = item.get("text", item.get("input_text", item.get("question", "")))
                prompts.append(str(text_content))
            return prompts
        # If template exists, format it
        return [self.template_string.format(**item) for item in batch_items]

def test_base_prompt_builder_init():
    builder_no_template = ConcreteTestPromptBuilder()
    assert builder_no_template.template_string is None
    assert builder_no_template.handler_args == {}

    template = "This is a {key}."
    args = {"arg1": "value1"}
    builder_with_template_args = ConcreteTestPromptBuilder(template_string=template, handler_args=args)
    assert builder_with_template_args.template_string == template
    assert builder_with_template_args.handler_args == args

def test_format_single_prompt_success():
    builder = ConcreteTestPromptBuilder() # Using the simple concrete implementation
    template = "Hello, {name}! You are {age}."
    data = {"name": "World", "age": 100}
    prompt = builder._format_single_prompt(template, data)
    assert prompt == "Hello, World! You are 100."

@patch('builtins.print') # Mock print as BasePromptBuilder uses it for warnings
def test_format_single_prompt_key_error(mock_print):
    builder = ConcreteTestPromptBuilder()
    template = "Hello, {name}! You are {missing_key}."
    data = {"name": "World", "age": 100}
    fallback = "Fallback text"
    prompt = builder._format_single_prompt(template, data, fallback_text=fallback)
    assert prompt == fallback
    # Check that the warning print was called with the expected content
    # The exact error message from .format() can vary slightly, so we check for key parts.
    # Example: "KeyError: 'missing_key'" or similar.
    # The BasePromptBuilder's _format_single_prompt prints a custom warning.
    mock_print.assert_any_call("Warning: Prompt formatting missing key: 'missing_key'. Template: 'Hello, {name}! You are {missing_key}.'. Data keys: ['name', 'age']")


@patch('builtins.print') # Mock print for warnings
def test_format_single_prompt_generic_error(mock_print):
    builder = ConcreteTestPromptBuilder()
    template = "Value: {value:%Y-%m-%d}" # Invalid format spec for an integer
    data = {"value": 123}
    fallback = "Generic fallback"
    prompt = builder._format_single_prompt(template, data, fallback_text=fallback)
    assert prompt == fallback
    # The actual exception message for invalid format spec with int is "Invalid format specifier..."
    # We check if our custom warning prefix is present, and part of the template.
    # The original test had a very specific error message from string.format, which can be fragile.
    # Making it more general to check if print was called due to an error.
    # The BasePromptBuilder catches general Exception as ex.
    # We assert that *a* warning was printed. The exact message from the caught exception might vary.
    # For a more precise test, we could inspect the *content* of the mock_print call.
    
    # Check that print was called (indicating a warning was issued)
    mock_print.assert_called()
    # Optionally, check if parts of the expected warning message are present in any of the calls
    found_warning = False
    for call_args in mock_print.call_args_list:
        args, _ = call_args
        if args and "Warning: Generic prompt formatting error:" in args[0] and template in args[0]:
            found_warning = True
            break
    assert found_warning, f"Expected generic formatting error warning not printed. Calls: {mock_print.call_args_list}"


# --- Tests for TemplateBasedPromptBuilder ---
def test_template_based_prompt_builder_with_template():
    template = "Question: {q}\nAnswer:"
    builder = TemplateBasedPromptBuilder(template_string=template)
    batch_items = [{"q": "What is 2+2?"}, {"q": "Capital of France?"}]
    prompts = builder.build_prompts(batch_items)
    assert prompts == ["Question: What is 2+2?\nAnswer:", "Question: Capital of France?\nAnswer:"]

def test_template_based_prompt_builder_no_template():
    builder = TemplateBasedPromptBuilder() # No template_string
    batch_items = [
        {"input_text": "Text 1", "question": "Q1"}, 
        {"question": "Q2"},
        {"input_text": "Text 3"},
        {"other_field": "data"} # This item should result in an empty string
    ]
    # Expected fallback order: "input_text", then "question", then ""
    expected_prompts = ["Text 1", "Q2", "Text 3", ""]
    prompts = builder.build_prompts(batch_items)
    assert prompts == expected_prompts

# --- Tests for MMLUPromptBuilder ---
def test_mmlu_prompt_builder_default_template():
    builder = MMLUPromptBuilder() # Uses DEFAULT_TEMPLATE
    item = {
        "question": "1+1=?", 
        "choices_formatted_str": "A) 1\nB) 2\nC) 3", 
        "subject": "math" # Subject is used in the default template from concrete_prompt_builders
    }
    prompts = builder.build_prompts([item])
    # The default template in MMLUPromptBuilder is:
    # "Question: {question}\nChoices:\n{choices_formatted_str}\nAnswer: "
    # It does not include {subject} by default.
    # However, if the prompt_template were to include it, the handler_args should provide it.
    # Let's assume the default template provided in your code doesn't use subject.
    # The MMLUPromptBuilder in concrete_prompt_builders.py has:
    # DEFAULT_TEMPLATE = "Question: {question}\nChoices:\n{choices_formatted_str}\nAnswer: "
    # This does not use {subject}.
    expected = "Question: 1+1=?\nChoices:\nA) 1\nB) 2\nC) 3\nAnswer: "
    assert prompts[0] == expected

def test_mmlu_prompt_builder_custom_template_and_choices_generation():
    template = "Sub: {subject}\nQ: {question}\n{choices_formatted_str}\nChoices by letter: A){choice_A} B){choice_B}\nA: "
    # MMLUPromptBuilder receives handler_args which might include default_subject.
    # Let's test if it correctly uses the subject from the item if available.
    builder = MMLUPromptBuilder(template_string=template, handler_args={"default_subject": "general"})
    item_with_choices_list = {
        "question": "Best color?", 
        "choices": ["Red", "Blue"], # This will be processed
        "subject": "colors"
    }
    prompts = builder.build_prompts([item_with_choices_list])
    expected = "Sub: colors\nQ: Best color?\nA) Red\nB) Blue\nChoices by letter: A)Red B)Blue\nA: "
    assert prompts[0] == expected
    
    # Test item with pre-formatted choices string (choices list should be ignored if choices_formatted_str exists)
    item_preformatted = {
        "question": "Best fruit?",
        "choices_formatted_str": "X. Apple\nY. Banana", # This should be used
        "choices": ["ignored_A", "ignored_B"], # This should be ignored
        "subject": "fruits"
    }
    prompts_preformatted = builder.build_prompts([item_preformatted])
    # The custom template expects choice_A, choice_B, but if choices_formatted_str is used,
    # the individual choice_X fields might not be populated from the list.
    # The MMLUPromptBuilder populates choice_X if it generates choices_formatted_str.
    # If choices_formatted_str is provided, it doesn't re-parse it to populate choice_X.
    # Let's adjust the template or the expectation.
    # For this test, let's assume the template only uses choices_formatted_str when it's complex.
    # Or, we test that choice_A, choice_B are NOT populated from choices_formatted_str.

    template_simple = "Sub: {subject}\nQ: {question}\n{choices_formatted_str}\nA: "
    builder_simple_template = MMLUPromptBuilder(template_string=template_simple, handler_args={"default_subject": "general"})
    prompts_preformatted_simple = builder_simple_template.build_prompts([item_preformatted])
    expected_preformatted_simple = "Sub: fruits\nQ: Best fruit?\nX. Apple\nY. Banana\nA: "
    assert prompts_preformatted_simple[0] == expected_preformatted_simple


# --- Tests for TranslationPromptBuilder ---
def test_translation_prompt_builder_default_template():
    # handler_args are passed to the constructor of the builder
    handler_args = {"source_lang_name": "English", "target_lang_name": "French",
                    "source_lang_code": "en", "target_lang_code": "fr"} # Add codes as they are in handler_args
    builder = TranslationPromptBuilder(handler_args=handler_args) # No template_string, uses default
    item = {"input_text": "Hello"}
    prompts = builder.build_prompts([item])
    # Default template in TranslationPromptBuilder is:
    # "Translate the following text from {source_lang_name} to {target_lang_name}: {input_text}"
    expected = "Translate the following text from English to French: Hello"
    assert prompts[0] == expected

def test_translation_prompt_builder_custom_template():
    template = "Translate from {source_lang_code} to {target_lang_code}: {input_text}"
    handler_args = {
        "source_lang_code": "en", "target_lang_code": "fr",
        "source_lang_name": "English", "target_lang_name": "French" 
    }
    builder = TranslationPromptBuilder(template_string=template, handler_args=handler_args)
    item = {"input_text": "Good morning"}
    prompts = builder.build_prompts([item])
    expected = "Translate from en to fr: Good morning"
    assert prompts[0] == expected

# --- Tests for GLUE PromptBuilders ---
def test_glue_sst2_prompt_builder():
    builder_default = GlueSST2PromptBuilder()
    item = {"input_text": "This movie is great."}
    prompt_default = builder_default.build_prompts([item])[0]
    # Default template: "Classify the sentiment...Review: \"{input_text}\"\nSentiment:"
    assert "Classify the sentiment of the following review as 'positive' or 'negative'." in prompt_default
    assert "Review: \"This movie is great.\"" in prompt_default
    assert prompt_default.endswith("Sentiment:")

    custom_template = "Review: {input_text}\nIs it good or bad?"
    builder_custom = GlueSST2PromptBuilder(template_string=custom_template)
    prompt_custom = builder_custom.build_prompts([item])[0]
    assert prompt_custom == "Review: This movie is great.\nIs it good or bad?"

def test_glue_mrpc_prompt_builder():
    builder = GlueMRPCPromptBuilder()
    item = {"input_text_pair1": "Sentence A.", "input_text_pair2": "Sentence B."}
    prompt = builder.build_prompts([item])[0]
    # Default template: "Are the following two sentences paraphrases? Answer ONLY with 'yes' or 'no'.\nSentence 1: \"{input_text_pair1}\"\nSentence 2: \"{input_text_pair2}\"\nAnswer:"
    assert "Are the following two sentences paraphrases? Answer ONLY with 'yes' or 'no'." in prompt
    assert "Sentence 1: \"Sentence A.\"" in prompt
    assert "Sentence 2: \"Sentence B.\"" in prompt
    assert prompt.endswith("Answer:")

def test_glue_stsb_prompt_builder():
    builder = GlueSTSBPromptBuilder()
    item = {"input_text_pair1": "Plane taking off.", "input_text_pair2": "Airplane ascending."}
    prompt = builder.build_prompts([item])[0]
    # Default template: "Rate the similarity ... numerical score.\nSentence 1: \"{input_text_pair1}\"\nSentence 2: \"{input_text_pair2}\"\nSimilarity Score:"
    assert "Rate the similarity of the following two sentences on a scale of 0.0 to 5.0. Provide ONLY the numerical score." in prompt
    assert "Sentence 1: \"Plane taking off.\"" in prompt
    assert "Sentence 2: \"Airplane ascending.\"" in prompt
    assert prompt.endswith("Similarity Score:")


# --- Tests for PromptBuilderFactory ---
def test_prompt_builder_factory_get_default():
    builder = PromptBuilderFactory.get_builder() # No type, no template
    assert isinstance(builder, TemplateBasedPromptBuilder)
    assert builder.template_string is None

def test_prompt_builder_factory_get_template_based_with_template():
    template = "My template: {data}"
    builder = PromptBuilderFactory.get_builder(builder_type="template_based", prompt_template=template)
    assert isinstance(builder, TemplateBasedPromptBuilder)
    assert builder.template_string == template

def test_prompt_builder_factory_get_mmlu():
    template = "MMLU Q: {question}"
    args = {"dataset_name": "mmlu_test", "default_subject": "knowledge"} # MMLUPromptBuilder expects default_subject in handler_args
    builder = PromptBuilderFactory.get_builder(builder_type="mmlu", prompt_template=template, handler_args=args)
    assert isinstance(builder, MMLUPromptBuilder)
    assert builder.template_string == template
    assert builder.handler_args.get("dataset_name") == "mmlu_test"

def test_prompt_builder_factory_get_translation():
    args = {"source_lang_name": "EN", "target_lang_name": "DE", "source_lang_code": "en", "target_lang_code": "de"}
    builder = PromptBuilderFactory.get_builder(builder_type="translation", handler_args=args)
    assert isinstance(builder, TranslationPromptBuilder)
    assert builder.handler_args.get("source_lang_name") == "EN"

@pytest.mark.parametrize("builder_key, expected_class", [
    ("glue_sst2_prompt", GlueSST2PromptBuilder),
    ("glue_mrpc_prompt", GlueMRPCPromptBuilder),
    ("glue_stsb_prompt", GlueSTSBPromptBuilder),
])
def test_prompt_builder_factory_get_glue_builders(builder_key, expected_class):
    builder = PromptBuilderFactory.get_builder(builder_type=builder_key)
    assert isinstance(builder, expected_class)

def test_prompt_builder_factory_unknown_type_fallback():
    # The logger is at the module level in prompt_builder_factory.py
    with patch('src.benchmark.prompting.prompt_builder_factory.logger') as mock_logger_module_level:
        builder = PromptBuilderFactory.get_builder(builder_type="unknown_nonexistent_type")
        assert isinstance(builder, TemplateBasedPromptBuilder) # Falls back to default
        mock_logger_module_level.warning.assert_called_with(
            "No prompt builder registered for type 'unknown_nonexistent_type'. Falling back to TemplateBasedPromptBuilder."
        )

# Example of registering a new builder (optional test, or for internal testing of register_builder)
class MyCustomBuilderForTest(BasePromptBuilder): # Renamed to avoid conflict
    def build_prompts(self, batch_items: list[dict[str, Any]]) -> list[str]:
        return ["custom_test_prompt"] * len(batch_items)

def test_prompt_builder_factory_register_and_get_custom():
    original_registry = PromptBuilderFactory._BUILDER_REGISTRY.copy() # Save original
    try:
        PromptBuilderFactory.register_builder("my_custom_test_builder", MyCustomBuilderForTest)
        builder = PromptBuilderFactory.get_builder("my_custom_test_builder")
        assert isinstance(builder, MyCustomBuilderForTest)
    finally:
        PromptBuilderFactory._BUILDER_REGISTRY = original_registry # Restore