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
    logger.info(f"Executing example_custom_batch_processor with script_args: {script_args}")
    
    input_texts = batch.get("input_text", []) # Assuming normalized field name
    original_labels = batch.get("target_text", [None] * len(input_texts)) # Or "label_index" etc.

    if not input_texts:
        return [], []
    
    prompts_to_model = []
    if advanced_args.get("prompt_template"):
        # Basic template filling example if no separate PromptBuilder is used
        template = advanced_args["prompt_template"]
        for text in input_texts:
            prompts_to_model.append(template.format(input_text=text))
    else:
        prompts_to_model = input_texts # Assume input_text is already the prompt

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