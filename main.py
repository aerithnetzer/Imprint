from imprint import Imprint
from pytesseract import Output
import os
import glob
import argparse

parser = argparse.ArgumentParser(
    prog="Imprint", description="Imprint takes as input a folder path"
)
parser.add_argument("-i", "--input")
parser.add_argument(
    "-o",
    "--output",
)
parser.add_argument("-b", "--benchmark", action="store_true")
parser.add_argument("-a", "--alias")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-r", "--recursive", action="store_true")
args = parser.parse_args()


def main():
    valid_image_suffixes = ["jpg", "jp2", "png", "tiff"]
    image_paths = []
    input: str = str(args.input)
    recursive: bool = bool(args.recursive)
    alias: bool = bool(args.alias)
    print(input, recursive, alias)
    norm_image_paths = []
    if os.path.isdir(input):
        image_paths = sorted(glob.glob(f"{input}/**/*", recursive=True))
        print(image_paths)
        if recursive:
            for i in image_paths:
                ext = str(os.path.splitext(i)[-1]).replace(".", "")
                if ext in valid_image_suffixes:
                    norm_image_paths.append(i)

            print(norm_image_paths)
        else:
            image_paths = sorted(glob.glob(f"{input}/*"))
            for i in image_paths:
                ext = str(os.path.splitext(i)[-1]).replace(".", "")
                if ext in valid_image_suffixes:
                    norm_image_paths.append(i)

            print(norm_image_paths)
            print(image_paths)
    elif os.path.isfile(input):
        image_paths = [input]
    else:
        print("Input path does not exist or is invalid.")
        return

    i = Imprint(
        norm_image_paths,
        use_ollama=False,
        transformer_model="",  # If True, use the transformer model defined in the transformer_model parameter. If False will default to pytesseract
        use_hf=False,
        benchmark=args.benchmark,
    )

    image = i.load("./tests/859.jpg")
    image = i.correct_rotation(image)
    i.show(image)


if __name__ == "__main__":
    main()
