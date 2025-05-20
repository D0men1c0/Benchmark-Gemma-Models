import re
from typing import Dict, Any, Tuple, List

import torch
from utils.logger import setup_logger

logger = setup_logger(__name__)

def example_custom_batch_processor(
    batch: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: str,
    advanced_args: Dict[str, Any],
    script_args: Dict[str, Any]
) -> Tuple[List[Any], List[Any]]:
    """
    Processes a batch for a custom task.
    Must return a tuple: (predictions, labels)
    :param batch: The input batch of data.
    :param model: The model to use for inference.
    :param tokenizer: The tokenizer to use for processing text.
    :param device: The device to run the model on.
    :param advanced_args: Advanced arguments for model inference.
    :param script_args: Additional arguments passed from the YAML configuration.
    :return: A tuple of (predictions, labels).
    """
    logger.info(f"Executing example_custom_batch_processor. Batch keys: {list(batch.keys())}. Script_args: {script_args}")

    input_texts = batch.get("text_content", [])
    original_labels_data = batch.get("category_label", [])

    logger.info(f"Number of input_texts obtained: {len(input_texts)}")

    if input_texts:
        original_labels = original_labels_data
    else:
        original_labels = []

    logger.info(f"Original labels obtained (first 5 if many, or all if less): {original_labels[:5]}")

    if not input_texts:
        logger.warning("example_custom_batch_processor: input_texts is empty, returning ([], [])")
        return [], []
    
    prompts_to_model = []
    prompt_template_from_yaml = advanced_args.get("prompt_template")
    if prompt_template_from_yaml:
        template = prompt_template_from_yaml
        for text in input_texts:
            prompts_to_model.append(template.format(input_text=text))
    else:
        prompts_to_model = input_texts

    # --- 2. Model Inference ---
    inputs = tokenizer(prompts_to_model, return_tensors="pt", padding=True, truncation=True,
                       max_length=advanced_args.get("generate_max_length", 512)).to(device)
    
    generate_kwargs = {
        "max_new_tokens": advanced_args.get("max_new_tokens", 20),
        "pad_token_id": tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    }
    if "num_beams" in advanced_args: generate_kwargs["num_beams"] = advanced_args["num_beams"]
    if "do_sample" in advanced_args: generate_kwargs["do_sample"] = advanced_args["do_sample"]
    # Add more generation params from advanced_args as needed...

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generate_kwargs)
    
    # Decode only the newly generated tokens
    input_ids_len = inputs.input_ids.shape[1]
    newly_generated_ids = generated_ids[:, input_ids_len:]
    predictions = tokenizer.batch_decode(newly_generated_ids, skip_special_tokens=True)
    

    logger.info(f"  Custom handler generated: {predictions[:2]}...") # Log a sample
    
    return predictions, original_labels

def creative_elaboration_processor(
    batch: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: str,
    advanced_args: Dict[str, Any],
    script_args: Dict[str, Any] 
) -> Tuple[List[str], List[str]]:
    """
    Prepares prompts for a creative elaboration task.
    :param batch: The input batch of data.
    :param model: The model to use for inference.
    :param tokenizer: The tokenizer to use for processing text.
    :param device: The device to run the model on.
    :param advanced_args: Advanced arguments for model inference.
    :param script_args: Additional arguments passed from the YAML configuration.
    :return: A tuple of (prompts_to_model, cleaned_input_themes).
    """
    logger.info(f"Executing prepare_creative_elaboration_prompts. Batch keys: {list(batch.keys())}. Script_args: {script_args}")

    raw_input_texts = batch.get("text_content", [])
    
    if not raw_input_texts:
        logger.warning("prepare_creative_elaboration_prompts: raw_input_texts is empty, returning ([], [])")
        return [], []

    cleaned_input_themes = []
    for text in raw_input_texts:
        cleaned_text = re.sub(r"Sample (train|validation|test) text \d+:\s*", "", text, 1)
        cleaned_input_themes.append(cleaned_text.strip())
    
    logger.info(f"  Cleaned input themes (first 2 if many): {cleaned_input_themes[:2]}")

    prompts_to_model = []
    template = advanced_args.get("prompt_template", "Expand on this idea with a creative paragraph:\nIdea: {theme}\n\nCreative Paragraph:") # Default se non specificato
    logger.info(f"  Using prompt template for creative elaboration: \"{template}\"")

    for theme_text in cleaned_input_themes:
        prompts_to_model.append(template.format(theme=theme_text))

    if not prompts_to_model:
        logger.warning("prepare_creative_elaboration_prompts: No prompts were generated.")
        return [], cleaned_input_themes if cleaned_input_themes else []
        
    logger.info(f"  Prepared {len(prompts_to_model)} prompts.")
    return prompts_to_model, cleaned_input_themes