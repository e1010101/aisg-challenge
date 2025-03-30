import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os # For checking file path

# --- Model Loading Function (Recommended to call once) ---
def load_captioning_model(model_name="fancyfeast/llama-joycaption-alpha-two-hf-llava", device_map="auto"):
    """
    Loads the Llava model and processor.

    Args:
        model_name (str): The Hugging Face model identifier.
        device_map (str or int): Device placement strategy ('auto', 0, 'cpu', etc.).
                                 'auto' is generally recommended for multi-GPU or CPU fallback.

    Returns:
        tuple: (processor, model, device) or (None, None, None) if loading fails.
               'device' is the primary device the model was loaded onto.
    """
    print(f"Loading model '{model_name}'...")
    try:
        # Determine the appropriate dtype (bfloat16 if supported, else float16)
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using bfloat16.")
        else:
            dtype = torch.float16
            print("Using float16 (bfloat16 not supported or no CUDA).")

        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True, # Helps with memory if model is large
            device_map=device_map # Automatically handle device placement
        )
        model.eval() # Set the model to evaluation mode (disables dropout etc.)

        # Determine the device the model is primarily on after device_map
        # This handles cases where device_map places parts on CPU/multiple GPUs
        try:
             device = model.device
        except AttributeError:
             # Handle models potentially split across devices by device_map more gracefully
             # A simple approach is to target the device of the first parameter
             device = next(model.parameters()).device
             print(f"Model potentially split across devices. Using device of first parameter: {device}")


        print(f"Model '{model_name}' loaded successfully on device: {device}")
        return processor, model, device

    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed error trace
        return None, None, None

# --- Image Captioning Function ---
def generate_caption(
    image_path: str,
    processor: AutoProcessor,
    model: LlavaForConditionalGeneration,
    device: torch.device or str, # Pass the device determined during loading
    prompt: str = "Write a long descriptive caption for this image in a formal tone.",
    system_prompt: str = "You are a helpful image captioner.",
    max_new_tokens: int = 300,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.9
) -> str | None:
    """
    Generates a caption for a given image using a pre-loaded Llava model.

    Args:
        image_path (str): Path to the input image file.
        processor (AutoProcessor): The pre-loaded processor for the model.
        model (LlavaForConditionalGeneration): The pre-loaded Llava model.
        device (torch.device or str): The device the model/inputs should be on.
        prompt (str, optional): The user prompt for captioning.
        system_prompt (str, optional): The system message for the conversation.
        max_new_tokens (int, optional): Max tokens for the generated caption.
        do_sample (bool, optional): Whether to use sampling.
        temperature (float, optional): Sampling temperature.
        top_p (float, optional): Nucleus sampling probability.

    Returns:
        str: The generated caption, or None if an error occurs.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return None

    try:
        # Load image
        image = Image.open(image_path)
        # Ensure image is in RGB format, as expected by many vision models
        if image.mode != "RGB":
            image = image.convert("RGB")
            print(f"Converted image '{os.path.basename(image_path)}' to RGB.")

    except Exception as e:
        print(f"Error loading or converting image '{image_path}': {e}")
        return None

    # Disable gradient calculations - crucial for inference
    with torch.no_grad():
        try:
            # Build the conversation following the required format
            convo = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Format the conversation string using the processor's template
            convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            if not isinstance(convo_string, str):
                 # Handle potential issues with template application if necessary
                 raise ValueError("Failed to apply chat template correctly. Expected string output.")


            # Process the inputs (text prompt and image)
            inputs = processor(
                text=[convo_string], # Note: text expects a list
                images=[image],      # Note: images expects a list
                return_tensors="pt"
            ).to(device) # Move inputs to the same device as the model

            # Ensure pixel values have the same dtype as the model
            inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)

            # Generate token IDs
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True, # Generally speeds up generation
                temperature=temperature,
                top_p=top_p,
                # suppress_tokens=None, # Optional: List of token IDs to suppress
                # top_k=None, # Usually top_p is used instead
            )[0] # Get the first generated sequence

            # Trim off the input prompt tokens from the generated sequence
            input_token_len = inputs['input_ids'].shape[1]
            generated_text_ids = generate_ids[input_token_len:]

            if len(generated_text_ids) == 0:
                 print("Warning: Model generated an empty sequence after trimming input.")
                 # Optionally return empty string or handle as error depending on need
                 return ""


            # Decode the generated token IDs back into a string
            caption = processor.tokenizer.decode(
                generated_text_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False # Per original code comment
            )

            return caption.strip() # Remove leading/trailing whitespace

        except Exception as e:
            print(f"Error during caption generation for '{os.path.basename(image_path)}': {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for detailed error trace
            return None