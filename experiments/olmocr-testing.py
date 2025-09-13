# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-to-text", model="allenai/olmOCR-7B-0825-FP8")
print(pipe("../test-images/35556036056489_00000002.jpg"))
