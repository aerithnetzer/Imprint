import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:0"

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
image1 = load_image("../test-images/35556036056489_00000001.jpg")
image2 = load_image("../test-images/35556036056489_00000002.jpg")
image3 = load_image("../test-images/35556036056489_00000003.jpg")

processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/Idefics3-8B-Llama3", torch_dtype=torch.bfloat16
).to(DEVICE)

# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty.",
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "And how about this image?"},
        ],
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
