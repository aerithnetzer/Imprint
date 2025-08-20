from imprint import Imprint
import glob


def main():
    image_paths = sorted(glob.glob("./test-images/*.jpg"))
    i = Imprint(
        image_paths,
        use_transformer=False,  # If True, use the transformer model defined in the transformer_model parameter. If False will default to pytesseract
        transformer_model="gemma3:12b",  # IGNORED IF USE_TRANSFORMER IS FALSE
    )
    i.infer()
    i.save("apollo-missions")


if __name__ == "__main__":
    main()
