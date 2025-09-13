import os
import glob
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:0"

# Load model and processor
processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/Idefics3-8B-Llama3", torch_dtype=torch.bfloat16
).to(DEVICE)

# Collect all images in test-images directory
image_files = sorted(glob.glob("./test-images/*.jpg"))

# Storage for OCR results
ocr_results = {}

for image_path in image_files:
    image = load_image(image_path)

    # Build a simple OCR instruction
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Extract all readable text from this image."},
            ],
        }
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate OCR output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Save result
    ocr_results[image_path] = generated_texts[0]

# Print all OCR results
for path, text in ocr_results.items():
    print(f"\n--- OCR for {path} ---\n{text}\n")
