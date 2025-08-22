from imprint import Imprint
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
parser.add_argument("-b", "--benchmark", action="benchmark_true")
parser.add_argument("-a", "--alias")
parser.add_argument("-v", "--verbose", action="verbose_true")
parser.add_argument("-r", "--recursive", action="recursive_true")
args = parser.parse_args()


def main():
    image_paths = sorted(glob.glob(f"./{args.directoryname}/*.jpg"))
    if os.path.isdir(args.input):
        if args.recursive:
            image_paths = f"{args.input}/**/.jpg"
        image_paths = sorted(glob.glob(f"{args.input}/*.jpg"))
    elif os.path.isfile(args.input):
        image_paths = [args.input]

    i = Imprint(
        image_paths,
        use_transformer=False,  # If True, use the transformer model defined in the transformer_model parameter. If False will default to pytesseract
        transformer_model="gemma3:12b",  # IGNORED IF USE_TRANSFORMER IS FALSE
        benchmark=args.benchmark,
    )
    i.infer()
    i.save("apollo-missions")


if __name__ == "__main__":
    main()
