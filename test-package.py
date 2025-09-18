from imprint import Imprint

p = Imprint()
img = p.load("./tests/859.jpg")
p.show(img)
img = p.deskew_with_hough(img)
p.show(img)
img = p.binarize(img)
p.show(img)
txt = p.ocr(
    img,
)
print(txt)
