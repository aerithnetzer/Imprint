# Imprint (Pre-Alpha State)

This code will need extensive refactoring in the future. It does do the job, but it could be better organized and include much more features.

## Feature Roadmap

- [ ] OCR via Huggingface API
- [x] OCR via Ollama
- [x] OCR via Tesseract
- [x] Image de-noising
- [x] Conversion from RGB to BW
- [ ] Background Removal
- [ ] De-skewing
- [ ] Cropping

Imprint is a document OCR pipeline manager that does the following:

Takes as input a list of pathlike objects.

It will then run image pre-processing functions using [opencv](https://github.com/opencv/opencv-python): denoising the image; and making the image black and white.

Currently, the OCR model utilizes [Ollama](https://github.com/ollama/ollama) API to call the OCR function.

## Quickstart

1. **Install dependencies:**

   ```bash
   pip install opencv-python matplotlib tqdm ollama
   ```

2. **Prepare your images:**  
   Place all document page images (e.g., PNG/JPG) in a directory.

## Usage

```python
from main import Imprint

image_paths = [
    "./test-images/page1.jpg",
    "./test-images/page2.jpg",
    # Add more pages as needed
]

i = Imprint(image_paths, use_transformer=True, transformer_model="gemma3:12b")
i.infer()  # Runs OCR pipeline on all images
i.save("test-images-output")  # Saves processed images and markdown to output_dir
```

## Right, But Why?

Vendors for OCR can be very expensive. This is meant to alleviate the cost of image pre-processing and OCR at Northwestern University Libraries. This program is designed to be run on anything from a jank laptop (using the pytesseract model instead of Ollama) to a high-end server (using Ollama's high-parameter models).

We find that using Gemma3:12B is sufficient, and benchmarks at 30s/page on an Apple M4 Pro Macbook.
