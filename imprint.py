from pathlib import Path
import io
from PIL import Image
import numpy as np
import logging
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datetime import datetime
import glob
from typing import ByteString, List, Tuple
import cv2 as cv
import matplotlib.pyplot as plt
import base64
import ollama
from tqdm import tqdm


class Imprint:
    def __init__(
        self,
        paths: List[str],
        use_ollama: bool,
        use_hf: bool,
        transformer_model: str,
        benchmark: bool,
    ):
        self.paths = paths
        self.images = [path for path in paths]
        self.use_ollama = use_ollama
        self.use_hf = use_hf
        self.transformer_model = transformer_model
        self.results = None
        self.benchmark = benchmark

    def _ocr(
        self,
        img,
        use_ollama: bool,
        use_hf: bool,
        transformer_model: str,
        benchmark: bool,
    ):
        success, encoded_image = cv.imencode(".png", img)
        b64_str = base64.b64encode(encoded_image).decode("utf-8")
        ocr_start = datetime.now()
        if use_hf:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image
            import io

            processor = TrOCRProcessor.from_pretrained(transformer_model)
            model = VisionEncoderDecoderModel.from_pretrained(transformer_model)

            image = Image.open(io.BytesIO(img)).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if benchmark:
                ocr_end = datetime.now()
                print(ocr_end - ocr_start)
                return text, ocr_end - ocr_start
            return text

        if use_ollama:
            system_prompt = (
                "You are an OCR extraction assistant. "
                "Do not add any commentary, explanation, or extra text. "
                "Only output the exact text found in the image, formatted as requested (markdown tables, footnotes, headers)."
            )
            prompt = "Extract the text from this image:\n\n"
            response = ollama.chat(
                model=transformer_model,
                options={
                    "seed": 42,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.25,
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64_str],
                    },
                ],
            )
            if benchmark:
                ocr_end = datetime.now()
                print(ocr_end - ocr_start)
                return response["message"]["content"], ocr_end - ocr_start
            return response["message"]["content"]
        else:
            try:
                ocr_start = datetime.now()
                import pytesseract
                import numpy as np

                # Decode image bytes to numpy array
                nparr = np.frombuffer(img, np.uint8)
                image = cv.imdecode(nparr, cv.IMREAD_COLOR)
                text = pytesseract.image_to_string(image)
                if benchmark:
                    ocr_end = datetime.now()
                    print(ocr_end - ocr_start)
                    return text, ocr_end - ocr_start
                else:
                    return text
            except ImportError:
                return "pytesseract not installed. Please install it for OCR without transformers."

    def _denoise(self, img):
        dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return dst

    def _background_removal(self, img, return_bytes=False, show=False, show_mask=False):
        """
        Remove background from an image using GrabCut.

        Args:
            img (str | Path | np.ndarray): Image path or numpy array.
            return_bytes (bool): If True, return PNG bytes instead of RGBA array.
            show (bool): If True, display the result in a window.
            show_mask (bool): If True, display the mask instead of the RGBA image.

        Returns:
            np.ndarray | bytes | None:
                RGBA numpy array if return_bytes=False,
                PNG bytes if return_bytes=True,
                None if image cannot be read.
        """
        # Load image if path was given
        if isinstance(img, (str, Path)):
            img = cv.imread(str(img))
            if img is None:
                logging.warning(f"Skipping {img} (unable to read)")
                return None
        elif not isinstance(img, np.ndarray):
            raise TypeError("img must be a path or a numpy.ndarray")

        height, width = img.shape[:2]
        margin = 0.05  # 5% margin
        x = int(width * margin)
        y = int(height * margin)
        rect = (x, y, width - 2 * x, height - 2 * y)

        # Create mask and models
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Apply GrabCut
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

        # Binary mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        # Convert to RGBA
        output_rgba = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
        output_rgba[:, :, 3] = mask2 * 255  # alpha channel

        if show_mask:
            cv.imshow("GrabCut Mask", mask2 * 255)
            cv.waitKey(0)
            cv.destroyAllWindows()
            return mask2  # return mask for inspection

        if show:
            cv.imshow("Background Removed", output_rgba)
            cv.waitKey(0)
            cv.destroyAllWindows()

        if return_bytes:
            image_pil = Image.fromarray(output_rgba)
            buf = io.BytesIO()
            image_pil.save(buf, format="PNG")
            return buf.getvalue()  # PNG bytes

        return output_rgba  # RGBA numpy array

    def _make_bw(self, img):
        (thresh, blackAndWhiteImage) = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        return blackAndWhiteImage

    def _deskew(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        coords = cv.findNonZero(img)
        angle = cv.minAreaRect(coords)[-1]
        angle = -angle
        # Normalize angle so small skews stay small
        if angle < -45:
            angle = 90 + angle

        (h, w) = img.shape[:2]

        M = cv.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        deskewed = cv.warpAffine(
            img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE
        )
        return deskewed

    def _array_to_base64(self, img):
        img = Image.fromarray(img)
        buff = io.BytesIO()
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    def infer(self) -> List[Tuple]:
        results = []
        for img in tqdm(self.images, desc="Processing pages"):
            if img is not None:
                img = cv.imread(img)
                dns = self._denoise(img)
                bw_img = self._make_bw(dns)
                removed_background = self._background_removal(
                    bw_img, return_bytes=False
                )
                deskewed = self._deskew(removed_background)
                ocr_result = self._ocr(
                    deskewed,
                    use_ollama=self.use_ollama,
                    use_hf=self.use_hf,
                    transformer_model=self.transformer_model,
                    benchmark=self.benchmark,
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
                # md_file.write(f"# Page {idx + 1}\n\n")
                # md_file.write(f"![Original]({os.path.basename(orig_path)})\n\n")
                # md_file.write(f"![Denoised]({os.path.basename(dns_path)})\n\n")
                # md_file.write(f"![Black & White]({os.path.basename(bw_path)})\n\n")
                md_file.write(f"{ocr_result}\n\n")
        print(f"Saved markdown and images to {output_dir}")
