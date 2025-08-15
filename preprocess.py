
import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
import imghdr
import argparse


def is_valid_image(filepath):
    """Check if an image file is valid and can be opened."""
    try:
        # Check file size
        if os.path.getsize(filepath) == 0:
            return False

        # Check file type
        img_type = imghdr.what(filepath)
        if img_type not in ['jpeg', 'png', 'bmp']:
            return False

        # Try to decode the image
        with open(filepath, 'rb') as f:
            img_array = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                return False

        return True
    except Exception:
        return False


def resize_image(img, target_size=(224, 224)):
    """Resize an image to the target size."""
    try:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Resize error: {str(e)}")
        return None


def process_image(img_path, min_size=(12, 12), target_size=(224, 224)):
    """
    Process a single image:
    - If smaller than min_size, delete it.
    - Otherwise, resize to target_size and overwrite.

    Returns:
    - 1: Success (resized)
    - 0: Deleted (too small)
    - -1: Failed
    """
    try:
        if not is_valid_image(img_path):
            print(f"Invalid image, removing: {img_path}")
            os.remove(img_path)
            return 0

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image, removing: {img_path}")
            os.remove(img_path)
            return 0

        h, w = img.shape[:2]
        if h < min_size[0] or w < min_size[1]:
            os.remove(img_path)
            return 0  # Deleted small image

        resized_img = resize_image(img, target_size)
        if resized_img is None:
            return -1

        # Overwrite the original image
        cv2.imwrite(img_path, resized_img)
        return 1  # Success

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return -1


def process_directory(directory, min_size=(12, 12), target_size=(224, 224)):
    """Process all images in a directory using multiple processes."""
    image_paths = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}

    print(f"Scanning directory: {directory}")
    for root, _, files in os.walk(directory):
        for filename in files:
            if os.path.splitext(filename.lower())[1] in valid_extensions:
                image_paths.append(os.path.join(root, filename))

    print(f"Found {len(image_paths)} total image files.")

    # Use multiprocessing
    n_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {n_processes} processes.")

    process_func = partial(process_image, min_size=min_size, target_size=target_size)

    with multiprocessing.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_func, image_paths),
            total=len(image_paths),
            desc="Preprocessing images"
        ))

    successful = results.count(1)
    deleted = results.count(0)
    failed = results.count(-1)

    return successful, deleted, failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Preprocessing Script")
    parser.add_argument('--directory', type=str, required=True, help='Target directory containing images.')
    parser.add_argument('--min_height', type=int, default=12, help='Minimum image height.')
    parser.add_argument('--min_width', type=int, default=12, help='Minimum image width.')
    parser.add_argument('--target_height', type=int, default=224, help='Target image height.')
    parser.add_argument('--target_width', type=int, default=224, help='Target image width.')
    args = parser.parse_args()

    min_size = (args.min_height, args.min_width)
    target_size = (args.target_height, args.target_width)

    print(f"Starting image processing...")
    print(f"Target Directory: {args.directory}")
    print(f"Minimum Size Threshold: {min_size[0]}x{min_size[1]}")
    print(f"Target Resize Size: {target_size[0]}x{target_size[1]}")

    successful, deleted, failed = process_directory(
        args.directory,
        min_size=min_size,
        target_size=target_size
    )

    print(f"\nProcessing Complete!")
    print(f"Successfully resized: {successful} images")
    print(f"Deleted small images: {deleted} images")
    print(f"Failed to process: {failed} images")