from pathlib import Path
import glob
from typing import ByteString, List, Tuple
import cv2 as cv
import matplotlib.pyplot as plt
import base64
import ollama
from tqdm import tqdm


class Imprint:
    def __init__(self, paths: List[str], use_transformer: bool, transformer_model: str):
        self.paths = paths
        self.images = [cv.imread(path) for path in paths]
        self.use_transformer = use_transformer
        self.transformer_model = transformer_model
        self.results = None

    def _ocr(self, img: ByteString, use_transformer: bool, transformer_model: str):
        b64_str = base64.b64encode(img).decode("utf-8")

        if use_transformer:
            system_prompt = (
                "You are an OCR extraction assistant. "
                "Do not add any commentary, explanation, or extra text. "
                "Only output the exact text found in the image, formatted as requested (markdown tables, footnotes, headers)."
            )
            prompt = "Extract the text from this image:\n\n"
            response = ollama.chat(
                model=transformer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64_str],
                    },
                ],
            )
            return response["message"]["content"]
        else:
            try:
                import pytesseract
                import numpy as np

                # Decode image bytes to numpy array
                nparr = np.frombuffer(img, np.uint8)
                image = cv.imdecode(nparr, cv.IMREAD_COLOR)
                text = pytesseract.image_to_string(image)
                return text
            except ImportError:
                return "pytesseract not installed. Please install it for OCR without transformers."

    def _denoise(self, img):
        dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return dst

    def _make_bw(self, img):
        (thresh, blackAndWhiteImage) = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        return blackAndWhiteImage

    def infer(self) -> List[Tuple]:
        results = []
        for img in tqdm(self.images, desc="Processing pages"):
            if img is not None:
                dns = self._denoise(img)
                bw_img = self._make_bw(dns)
                img_bytes = cv.imencode(".png", bw_img)[1].tobytes()
                ocr_result = self._ocr(
                    img_bytes,
                    use_transformer=self.use_transformer,
                    transformer_model=self.transformer_model,
                )
                results.append((img, dns, bw_img, ocr_result))
            else:
                results.append((None, None, None, "Image not loaded."))
        self.results = results
        return results

    def show(self):
        results = self.infer()
        for idx, (orig, dns, bw_img, ocr_result) in enumerate(results):
            plt.figure(figsize=(10, 8))
            plt.suptitle(f"Page {idx + 1}")
            plt.subplot(2, 2, 1)
            if orig is not None:
                plt.imshow(orig)
                plt.title("Original")
            else:
                plt.title("Original (Not loaded)")
                plt.axis("off")

            plt.subplot(2, 2, 2)
            if dns is not None:
                plt.imshow(dns)
                plt.title("Denoised")
            else:
                plt.title("Denoised (Not loaded)")
                plt.axis("off")

            plt.subplot(2, 2, 3)
            if bw_img is not None:
                plt.imshow(bw_img)
                plt.title("Black & White")
            else:
                plt.title("Black & White (Not loaded)")
                plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.axis("off")
            plt.title("OCR Result")
            plt.text(
                0.5, 0.5, ocr_result, fontsize=12, ha="center", va="center", wrap=True
            )

            plt.tight_layout()
            plt.show()

    def save(self, output_dir: str, output_md: str = "output.md"):
        import os

        if self.results is None:
            raise RuntimeError("Run .infer() before .save()")

        os.makedirs(output_dir, exist_ok=True)
        md_path = os.path.join(output_dir, output_md)
        with open(md_path, "w", encoding="utf-8") as md_file:
            for idx, (orig, dns, bw_img, ocr_result) in enumerate(self.results):
                page_prefix = f"page_{idx + 1}"
                # Save images
                orig_path = os.path.join(output_dir, f"{page_prefix}_original.png")
                dns_path = os.path.join(output_dir, f"{page_prefix}_denoised.png")
                bw_path = os.path.join(output_dir, f"{page_prefix}_bw.png")
                if orig is not None:
                    cv.imwrite(orig_path, orig)
                if dns is not None:
                    cv.imwrite(dns_path, dns)
                if bw_img is not None:
                    cv.imwrite(bw_path, bw_img)
                # Write markdown
                md_file.write(f"# Page {idx + 1}\n\n")
                md_file.write(f"![Original]({os.path.basename(orig_path)})\n\n")
                md_file.write(f"![Denoised]({os.path.basename(dns_path)})\n\n")
                md_file.write(f"![Black & White]({os.path.basename(bw_path)})\n\n")
                md_file.write(f"{ocr_result}\n\n")
        print(f"Saved markdown and images to {output_dir}")


def main():
    image_paths = sorted(glob.glob("./test-images/*.jpg"))
    i = Imprint(
        image_paths,
        use_transformer=False,
        transformer_model="gemma3:12b",  # IGNORED IF USE_TRANSFORMER IS FALSE
    )
    i.infer()
    i.save("apollo-missions")


if __name__ == "__main__":
    main()
