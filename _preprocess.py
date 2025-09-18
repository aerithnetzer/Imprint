#!/usr/bin/env python3
"""
Image Preprocessing Script
Performs deskewing, denoising, binarization, and other cleanup operations on images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import argparse
import os


def load_image(image_path):
    """Load image and convert to grayscale if needed."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return gray


def deskew_image(image):
    """Detect and correct skew in the image."""
    try:
        # Apply binary threshold to get better edge detection
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find edges
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        # Use Hough transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines[:20]:  # Use first 20 lines
                if len(line) >= 2:  # Make sure we have at least 2 values
                    rho, theta = line[0], line[1]
                else:
                    continue

                # Convert angle to degrees
                angle = np.degrees(theta)

                # Normalize angle to be between -45 and 45 degrees
                if angle > 45:
                    angle = angle - 90
                elif angle < -45:
                    angle = angle + 90

                angles.append(angle)

            # Calculate median angle
            if angles:
                median_angle = np.median(angles)
                print(f"Detected skew angle: {median_angle:.2f} degrees")

                # Only correct if angle is significant (more than 0.5 degrees)
                if abs(median_angle) > 0.5:
                    # Rotate image to correct skew (negative angle to correct)
                    (h, w) = image.shape
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(
                        center, -median_angle, 1.0
                    )
                    deskewed = cv2.warpAffine(
                        image,
                        rotation_matrix,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    return deskewed
                else:
                    print("Skew angle too small, no correction needed")

        print("No significant skew detected")
        return image

    except Exception as e:
        print(f"Error in deskewing: {e}")
        print("Skipping deskew step")
        return image


def denoise_image(image, strength="light"):
    """Apply denoising filters to clean up the image.

    Args:
        image: Input grayscale image
        strength: 'none', 'light', 'medium', 'heavy'
    """
    if strength == "none":
        return image
    elif strength == "light":
        # Very gentle denoising
        denoised = cv2.bilateralFilter(image, 3, 25, 25)
    elif strength == "medium":
        # Moderate denoising
        denoised = cv2.GaussianBlur(image, (1, 1), 0)
        denoised = cv2.bilateralFilter(denoised, 5, 50, 50)
    else:  # heavy
        # Stronger denoising (original settings)
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    return denoised


def binarize_image(image):
    """Convert image to binary (black and white) using adaptive thresholding."""
    # Apply Otsu's thresholding
    _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply adaptive thresholding for better results with varying lighting
    binary_adaptive = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Combine both methods (you can experiment with different combinations)
    # Here we use adaptive thresholding as it typically works better for documents
    return binary_adaptive


def enhance_contrast(image):
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced


def remove_borders(image, border_size=10):
    """Remove border artifacts by cropping."""
    h, w = image.shape
    return image[border_size : h - border_size, border_size : w - border_size]


def process_image(image_path, show_steps=True, denoise_strength="light"):
    """Main processing pipeline."""
    print(f"Processing image: {image_path}")

    # Load image
    original = load_image(image_path)
    print(f"Image loaded: {original.shape}")

    # Store intermediate results
    steps = [("Original", original)]

    # Step 1: Enhance contrast
    contrast_enhanced = enhance_contrast(original)
    steps.append(("Contrast Enhanced", contrast_enhanced))

    # Step 2: Deskew
    deskewed = deskew_image(contrast_enhanced)
    steps.append(("Deskewed", deskewed))

    # Step 3: Denoise
    denoised = denoise_image(deskewed, strength=denoise_strength)
    steps.append((f"Denoised ({denoise_strength})", denoised))

    # Step 4: Remove borders
    no_borders = remove_borders(denoised)
    steps.append(("Borders Removed", no_borders))

    # Step 5: Binarize
    binary = binarize_image(no_borders)
    steps.append(("Binarized", binary))

    if show_steps:
        display_results(steps)

    return binary


def display_results(steps):
    """Display original and processed images side by side."""
    n_steps = len(steps)
    cols = 3
    rows = (n_steps + cols - 1) // cols

    plt.figure(figsize=(15, 5 * rows))

    for i, (title, img) in enumerate(steps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_result(image, output_path):
    """Save the processed image."""
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process and clean up images")
    parser.add_argument("input_path", help="Path to input image")
    parser.add_argument("-o", "--output", help="Path to save processed image")
    parser.add_argument(
        "--no-display", action="store_true", help="Don't display processing steps"
    )
    parser.add_argument(
        "--denoise",
        choices=["none", "light", "medium", "heavy"],
        default="light",
        help="Denoising strength (default: light)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' not found")
        return

    try:
        # Process the image
        result = process_image(
            args.input_path,
            show_steps=not args.no_display,
            denoise_strength=args.denoise,
        )

        # Save result if output path specified
        if args.output:
            save_result(result, args.output)
        else:
            # Auto-generate output filename
            base, ext = os.path.splitext(args.input_path)
            output_path = f"{base}_processed{ext}"
            save_result(result, output_path)

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    # If running without command line arguments, you can test with a specific image
    import sys

    if len(sys.argv) == 1:
        print("Usage examples:")
        print("python image_preprocess.py input_image.jpg")
        print("python image_preprocess.py input_image.jpg -o output_image.jpg")
        print("python image_preprocess.py input_image.jpg --no-display")
        print()
        print("Required libraries: opencv-python, numpy, matplotlib, scipy")
        print("Install with: pip install opencv-python numpy matplotlib scipy")
    else:
        main()
