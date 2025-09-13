import base64
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline

model_id = "allenai/olmOCR-7B-0825-FP8"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="cuda")

pipe = pipeline(
    "image-to-text",
    model=model,
    tokenizer=processor.tokenizer,
    image_processor=processor.image_processor,
    device=0,
)

# Read and encode image
image_path = (
    "/projects/p32234/projects/aerith/Imprint/test-images/35556036056489_00000002.jpg"
)
with open(image_path, "rb") as f:
    image_bytes = f.read()

# Encode → decode so you simulate having a base64 string
image_b64 = base64.b64encode(image_bytes).decode("utf-8")
decoded_bytes = base64.b64decode(image_b64)

# Call pipeline with bytes instead of path
result = pipe(
    {"image": decoded_bytes, "prompt": "Please transcribe the text in this image."}
)

print(result[0]["generated_text"])
