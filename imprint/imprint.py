# TODO:
# This needs more extensive documentation
# I also think that we should not have so many arguments.
# Perhaps we do something where each step is a method call. So that the user does:
# Imprint.binarize.
# Imprint.remove_background.
# Imprint.ocr(method arguments about HF, Ollama, Tesseract, etc. can go here)
# That way we can tune hyperparameters more easily, and then wrap each step in a CLI at the end.
import base64
from datetime import datetime
import io
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
from loguru import logger
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import numpy as np
import ollama
from PIL import Image
import pytesseract
from scipy.stats import entropy
from skimage.measure import shannon_entropy
from tqdm import tqdm


class Imprint:
    def __init__(
        self,
        paths: list | None = None,
        use_ollama: bool = False,
        use_hf: bool = False,
        transformer_model: str = "",
        benchmark: bool = False,
    ):
        self.paths = paths or []
        self.images = [path for path in self.paths]
        self.use_ollama = use_ollama
        self.use_hf = use_hf
        self.transformer_model = transformer_model
        self.results = None
        self.benchmark = benchmark

    @staticmethod
    def ocr(img, method="tesseract", model=""):
        options = ["huggingface", "tesseract", "ollama", "marker"]
        if method.lower() not in options:
            logger.critical(
                f"{method} not found. Choose from 'ollama', 'tesseract' or 'huggingface'"
            )
            raise ValueError(f"Invalid OCR method: {method}")
        if method.lower() == "tesseract":
            return pytesseract.image_to_string(img)
        elif method.lower() == "marker":
            config = {"output_format": "html"}
            config_parser = ConfigParser(config)
            converter = PdfConverter(
                artifact_dict=create_model_dict(),
                config=config_parser.generate_config_dict(),
            )
            rendered = converter(img)
            text, _, images = text_from_rendered(rendered)
            return text
        elif method.lower() == "ollama":
            system_prompt = (
                "You are an OCR extraction assistant. "
                "Do not add any commentary, explanation, or extra text. "
                "Only output the exact text found in the image, formatted as requested (markdown tables, footnotes, headers). It is a matter of life or death that you do not repeat text."
            )
            prompt = "Extract the text from this image:\n\n"
            response = ollama.chat(
                model="gemma3:27b",
                options={
                    "seed": 42,
                    "temperature": 0.35,
                    "top_p": 0.95,
                    "top_k": 40,
                    "repetition_penalty": 50,
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img],
                    },
                ],
            )
            return response["message"]["content"]

    @staticmethod
    def deskew_with_hough(img) -> np.ndarray:
        """Deskew an image containing text/diagrams using Hough + entropy check fallback.

        Features:
        - Uses entropy of a text-line mask to decide between Hough and projection profile.
        - Rejects phantom skew if detected angle is too small or too inconsistent.

        Args:
            img (np.ndarray): Input OpenCV image (BGR or grayscale).
        Returns:
            np.ndarray: Deskewed OpenCV image.
        """
        # --- Convert to grayscale if needed ---
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Invert so text = white
        gray_inv = cv2.bitwise_not(gray)

        # --- Morphological filtering to enhance horizontal text lines ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        textline_mask = cv2.morphologyEx(gray_inv, cv2.MORPH_CLOSE, kernel)

        # --- Entropy of the mask (higher = more text-like structure) ---
        hist = cv2.calcHist([textline_mask], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        mask_entropy = entropy(hist, base=2)

        # Threshold empirically: ~3.5 works well
        use_hough = mask_entropy > 3.5

        best_angle = 0
        angles = []

        if use_hough:
            # Edge detection for Hough
            edges = cv2.Canny(textline_mask, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

            if lines is not None:
                for rho, theta in lines[:, 0]:
                    angle = (theta * 180 / np.pi) - 90
                    if -45 < angle < 45:  # only near-horizontal
                        angles.append(angle)

            if angles:
                best_angle = np.median(angles)
                # Consistency check: if angles too scattered, ignore
                if np.std(angles) > 5:
                    best_angle = 0
            else:
                use_hough = False  # fallback

        if not use_hough:
            # --- Projection profile fallback ---
            shift_range = np.arange(-15, 16)  # search ±15°
            scores = []
            for s in shift_range:
                M = cv2.getRotationMatrix2D(
                    (gray.shape[1] // 2, gray.shape[0] // 2), s, 1
                )
                rotated = cv2.warpAffine(
                    gray_inv,
                    M,
                    (gray.shape[1], gray.shape[0]),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                proj = np.sum(rotated, axis=1)
                scores.append(np.var(proj))
            best_angle = shift_range[np.argmax(scores)]

        # --- Confidence threshold: skip tiny rotations ---
        if abs(best_angle) < 1.0:  # less than 1 degree
            best_angle = 0

        # --- Rotate original image if needed ---
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    @staticmethod
    def binarize_or_gray(img, text_threshold=0.5, entropy_threshold=4.0, debug=False):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(
            gray, None, h=10, templateWindowSize=7, searchWindowSize=21
        )

        # --- Edge analysis for "textiness" ---
        edges = cv2.Canny(denoised, 50, 150)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            edges, connectivity=8
        )
        sizes = stats[1:, cv2.CC_STAT_AREA]  # skip background

        if len(sizes) == 0:
            logger.debug("No connected components found → returning grayscale")
            return gray, "diagram"

        small_components = np.sum(sizes < 300)
        ratio_small = small_components / (len(sizes) + 1e-5)

        # --- Entropy analysis ---
        entropy_val = shannon_entropy(edges)

        # --- Decision logic ---
        if len(sizes) < 2500:
            decision = "text"
            result = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
            )
        else:
            decision = "diagram"
            result = gray

        logger.debug(
            f"Decision={decision} | ratio_small={ratio_small:.3f} "
            f"(threshold={text_threshold}) | entropy={entropy_val:.3f} "
            f"(threshold={entropy_threshold}) | components={len(sizes)}"
        )

        return result

    def _ocr(
        self,
        img,
        use_ollama: bool,
        use_hf: bool,
        transformer_model: str,
        benchmark: bool,
    ):
        success, encoded_image = cv2.imencode(".png", img)
        b64_str = base64.b64encode(encoded_image).decode("utf-8")
        ocr_start = datetime.now()
        if use_hf:
            import io

            from PIL import Image
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

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
                "Only output the exact text found in the image, formatted as requested (markdown tables, footnotes, headers). Do not repeat text."
            )
            prompt = "Extract the text from this image:\n\n"
            response = ollama.chat(
                model=transformer_model,
                options={
                    "seed": 42,
                    "temperature": 0.35,
                    "top_p": 0.95,
                    "top_k": 40,
                    "repetition_penalty": 50,
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
                import numpy as np
                from PIL import Image
                import pytesseract

                print(type(img))
                img = Image.fromarray(img)
                # Decode image bytes to numpy array
                text = pytesseract.image_to_string(img)
                if benchmark:
                    ocr_end = datetime.now()
                    print(ocr_end - ocr_start)
                    return text, ocr_end - ocr_start
                else:
                    return text
            except ImportError:
                return "pytesseract not installed. Please install it for OCR without transformers."

    def _denoise(self, img):
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return dst

    def _background_removal(
        self,
        img,
        return_bytes: bool = False,
        show: bool = False,
        show_mask: bool = False,
        scale: float = 0.25,
        iterations: int = 3,
    ):
        """
        Remove background from an image using GrabCut with optimization (downscaling).

        Args:
            img (str | Path | np.ndarray): Image path or numpy array.
            return_bytes (bool): If True, return PNG bytes instead of RGBA array.
            show (bool): If True, display the result in a window.
            show_mask (bool): If True, display the mask instead of the RGBA image.
            scale (float): Resize factor for faster GrabCut (0.25 = 25% size).
            iterations (int): Number of GrabCut iterations (lower = faster).

        Returns:
            np.ndarray | bytes | None:
                RGBA numpy array if return_bytes=False,
                PNG bytes if return_bytes=True,
                None if image cannot be read.
        """
        # Load image if path was given
        if isinstance(img, (str, Path)):
            img = cv2.imread(str(img))
            if img is None:
                logging.warning(f"Skipping {img} (unable to read)")
                return None
        elif not isinstance(img, np.ndarray):
            raise TypeError("img must be a path or a numpy.ndarray")

        height, width = img.shape[:2]

        # Downscale for faster GrabCut
        if scale < 1.0:
            small = cv2.resize(img, (int(width * scale), int(height * scale)))
        else:
            small = img.copy()

        # Define rectangle with margin (relative to small image)
        margin = 0.05
        sw, sh = small.shape[1], small.shape[0]
        x = int(sw * margin)
        y = int(sh * margin)
        rect = (x, y, sw - 2 * x, sh - 2 * y)

        # Create mask and models
        mask_small = np.zeros(small.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Run GrabCut on small image
        cv2.grabCut(
            small,
            mask_small,
            rect,
            bgdModel,
            fgdModel,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )

        # Convert GrabCut output to binary mask
        mask_small = np.where((mask_small == 2) | (mask_small == 0), 0, 1).astype(
            "uint8"
        )

        # Upscale mask to original resolution
        mask = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST)

        # Convert original image to RGBA with alpha channel
        output_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        output_rgba[:, :, 3] = mask * 255

        # Debug options
        if show_mask:
            cv2.imshow("GrabCut Mask", mask * 255)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return mask

        if show:
            cv2.imshow("Background Removed", output_rgba)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if return_bytes:
            image_pil = Image.fromarray(output_rgba)
            buf = io.BytesIO()
            image_pil.save(buf, format="PNG")
            return buf.getvalue()

        return output_rgba

    def _make_bw(self, img):
        (thresh, blackAndWhiteImage) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return blackAndWhiteImage

    def _deskew(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(img)
        angle = cv2.minAreaRect(coords)[-1]
        angle = -angle
        # Normalize angle so small skews stay small
        if angle < -45:
            angle = 90 + angle

        (h, w) = img.shape[:2]

        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        deskewed = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed

    def _array_to_base64(self, img):
        img = Image.fromarray(img)
        buff = io.BytesIO()
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    def _remove_artifacts(
        self,
        img,
        nlm_strength=10,
        min_component_area=15,
        max_component_area=None,
        aspect_ratio_filter=None,
    ):
        """
        Advanced version with more control over parameters

        Args:
            image_path: Input image path
            output_path: Output path (optional)
            nlm_strength: NLM denoising strength (higher = more denoising)
            min_component_area: Minimum area to keep components
            max_component_area: Maximum area to keep components (None = no limit)
            aspect_ratio_filter: (min_ratio, max_ratio) to filter by aspect ratio
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Ensure proper data type
        print("Type of gray", gray.dtype)
        if gray.dtype == np.uint8:
            gray = gray.astype(np.uint8)
            # Apply NLM denoising
            denoised = cv2.fastNlMeansDenoising(
                img, None, h=nlm_strength, templateWindowSize=7, searchWindowSize=21
            )

            # Threshold
            _, binary = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Connected components analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                255 - binary, connectivity=8
            )

            # Create cleaned image
            cleaned = binary.copy()

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]

                should_remove = False

                # Area filter
                if area < min_component_area:
                    should_remove = True

                if max_component_area and area > max_component_area:
                    should_remove = True

                # Aspect ratio filter
                if aspect_ratio_filter and height > 0:
                    aspect_ratio = width / height
                    min_ratio, max_ratio = aspect_ratio_filter
                    if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                        should_remove = True

                if should_remove:
                    cleaned[labels == i] = 255

            return cleaned

    def load(self, image_path):
        return cv2.imread(image_path)

    def infer(self) -> List[Tuple]:
        results = []
        logger.info("Starting inference")
        for img in tqdm(self.images, desc="Processing pages"):
            if img is not None:
                logger.info(f"Processing {img}")
                logger.info("Reading file")
                img = self.load(img)
                logger.info(f"Fileshape: {img.shape}")
                logger.info("Denoising")
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

    @staticmethod
    def show(image):
        cv2.imshow("Imprint", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def save_image(image, file_path):
        cv2.imwrite(file_path, image)
        return True

    @staticmethod
    def remove_borders(img):
        # Make a copy
        out = img.copy()

        # Create mask required by floodFill (2 pixels larger)
        h, w = out.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Flood fill from each corner (in case some sides aren't connected)
        for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            if out[seed[1], seed[0]] == 0:  # only flood if pixel is black
                cv2.floodFill(out, mask, seedPoint=seed, newVal=255)

        return out

    def save_ocr(self, output_dir: str, output_md: str = "output.md"):
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
                    cv2.imwrite(orig_path, orig)
                if dns is not None:
                    cv2.imwrite(dns_path, dns)
                if bw_img is not None:
                    cv2.imwrite(bw_path, bw_img)
                # Write markdown
                md_file.write(f"# Page {idx + 1}\n\n")
                md_file.write(f"![Original]({os.path.basename(orig_path)})\n\n")
                md_file.write(f"![Denoised]({os.path.basename(dns_path)})\n\n")
                md_file.write(f"![Black & White]({os.path.basename(bw_path)})\n\n")
                md_file.write(f"{ocr_result}\n\n")
        print(f"Saved markdown and images to {output_dir}")

    class Pipelines:
        def __init__(self, img):
            self.img = img

        @staticmethod
        def process_image():
            pass

        @staticmethod
        def ocr():
            pass
