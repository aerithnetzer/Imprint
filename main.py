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
parser.add_argument("-b", "--benchmark", action="store_true")
parser.add_argument("-a", "--alias")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-r", "--recursive", action="store_true")
args = parser.parse_args()


def main():
    image_paths = []
    input: str = str(args.input)
    recursive: bool = bool(args.recursive)
    alias: bool = bool(args.alias)

    if os.path.isdir(input):
        if recursive:
            image_paths = sorted(glob.glob(f"{input}/**/*.jp2", recursive=True))
        else:
            image_paths = sorted(glob.glob(f"{input}/*.jp2"))
    elif os.path.isfile(input):
        image_paths = [input]
    else:
        print("Input path does not exist or is invalid.")
        return

    i = Imprint(
        image_paths,
        use_transformer=True,  # If True, use the transformer model defined in the transformer_model parameter. If False will default to pytesseract
        transformer_model="gemma3:12b-multigpu",  # IGNORED IF USE_TRANSFORMER IS FALSE
        benchmark=args.benchmark,
    )
    i.infer()
    i.save(args.output)


if __name__ == "__main__":
    main()
