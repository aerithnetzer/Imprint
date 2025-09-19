# Imprint (Pre-Alpha State)

Hello Northwestern Digitization team, welcome to Imprint, your new (hopefully) one-stop shop for digitization.
I have attempted to abstract as much of the intricacies of image transformation in python. At least to the best of my ability.
While we are working on this grant, I will be working on build automation and a CLI for you all so that it can be even easier to use.

Here is my current feature list, where I am open to suggestions or requests, because I like this sort of thing:

- [ ] API
  - [x] Load image
  - [x] Deskew
  - [x] Intelligent document binarization/grayscale
  - [x] Tesseract OCR
  - [x] Standalone Ollama LLM OCR
  - [x] Marker Document Understanding LLM OCR
  - [ ] HathiTrust (currently experimental in a standalone script)
- [ ] Pipelines
  - [x] Image processing
  - [ ] OCR
  - [ ] HathiTrust
  - [ ] ???
  - [ ] Profit
- [ ] CLI (To be done later)
- [ ] Figure out how the hell I publish a python package

## Prerequisites

You will want to familiarize yourself with the absolute basics of calling object-methods. If you want to use any LLM models, you will need to install [Ollama](https://ollama.com/). Feel free to contact me if you need assistance in setting up Ollama.

## Quickstart

If you are extremely impatient, you can get started with two lines of code

```python
from imprint import Imprint

Imprint.Pipelines.process_image("inputfilename.jpg", "outputfilename.jpg")
```

In the backend, it looks like this. You can find this code in the `cookbook` directory

```python
from imprint import Imprint

p = Imprint()  # Creates the "Imprint" class

f = "filename.jpg"
base_name = f.replace(".jpg", "")

# Step 1: Load original
img = p.load(f)

# Step 2: de-skew
img = p.deskew_with_hough(img)

# Step 3: This is kinda weird, and currently fine-tuned for use with NU's environmental impact statements
# Long story short, the programming will use some heuristics to detect if the image is a diagram or mostly text
# If the program thinks it is text, it will binarize, if it thinks it is a diagram, it will not.
img = p.binarize_or_gray(img)

p.save_image(img)
```
