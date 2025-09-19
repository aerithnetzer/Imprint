from ..imprint import Imprint

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


#
