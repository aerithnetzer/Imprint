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

result = pipe(
    {
        "image": "/projects/p32234/projects/aerith/Imprint/test-images/35556036056489_00000002.jpg",
        "prompt": "Please transcribe the text in this image.",
    }
)

print(result)
