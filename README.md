# Imprint

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
# Imprint
