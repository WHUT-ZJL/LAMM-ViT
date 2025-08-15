import os
import cv2
import random
import numpy as np
import torch
import io
from PIL import Image, ImageFilter
import warnings
from tqdm import tqdm
import h5py
import argparse
import time
import multiprocessing
import dlib
import gc

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DataAugmentation:
    """Implements various data augmentation methods."""

    def __init__(self, p_gaussian=0.1, p_jpeg=0.1, p_blur=0.1):
        self.p_gaussian = p_gaussian
        self.p_jpeg = p_jpeg
        self.p_blur = p_blur
        self.gaussian_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
        self.jpeg_qualities = [90, 80, 70, 60, 50]
        self.blur_sigmas = [1, 2, 3, 4, 5]

    def apply_gaussian_noise(self, image, level):
        """Add Gaussian noise."""
        image_np = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, level, image_np.shape)
        noisy_image = np.clip(image_np + noise, 0, 1) * 255
        return Image.fromarray(noisy_image.astype(np.uint8))

    def apply_jpeg_compression(self, image, quality):
        """Apply JPEG compression."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def apply_gaussian_blur(self, image, sigma):
        """Apply Gaussian blur."""
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def __call__(self, image):
        """Apply augmentations."""
        if random.random() < self.p_gaussian:
            image = self.apply_gaussian_noise(image, random.choice(self.gaussian_levels))
        if random.random() < self.p_jpeg:
            image = self.apply_jpeg_compression(image, random.choice(self.jpeg_qualities))
        if random.random() < self.p_blur:
            image = self.apply_gaussian_blur(image, random.choice(self.blur_sigmas))
        return image


class FeatureExtractor:
    """Handles facial landmark detection and feature extraction."""

    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            print("Downloading dlib facial landmark model...")
            import urllib.request
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            urllib.request.urlretrieve(url, model_path + ".bz2")
            import bz2
            print("Decompressing model file...")
            with open(model_path, 'wb') as new_file, bz2.BZ2File(model_path + ".bz2", 'rb') as file:
                new_file.write(file.read())
            print("Model ready.")
        self.predictor = dlib.shape_predictor(model_path)
        self.regions = {
            'left_eye': list(range(36, 42)), 'right_eye': list(range(42, 48)),
            'nose': list(range(27, 36)), 'upper_lip': list(range(48, 55)),
            'lower_lip': list(range(54, 60)) + [48, 60, 61, 62, 63, 64, 65, 66, 67],
            'mouth_corners': [48, 54],
            'face_outline': list(range(0, 17)), 'eyebrows': list(range(17, 27))
        }

    def detect_face_landmarks(self, image):
        """Detect facial landmarks using dlib."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        if not faces: return []
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = self.predictor(gray, face)
        return [(i, shape.part(i).x, shape.part(i).y) for i in range(68)]

    def generate_region_masks(self, image_shape, landmarks, patch_size=16):
        """Generate masks for 8 facial regions."""
        h, w = image_shape[:2]
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        region_masks = []

        for _, region_indices in self.regions.items():
            mask = np.zeros((h, w), dtype=np.float32)
            points = np.array([(x, y) for i, x, y in landmarks if i in region_indices], dtype=np.int32)
            if len(points) > 2:
                cv2.fillPoly(mask, [points], 1.0)

            # Downsample mask to patch level
            patch_mask = cv2.resize(mask, (num_patches_w, num_patches_h), interpolation=cv2.INTER_AREA)
            region_masks.append(patch_mask)

        return np.array(region_masks, dtype=np.float32)

    def extract_all_features(self, image, augmentation=None):
        """Extract all features from an image."""
        if augmentation:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_image = augmentation(pil_image)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        landmarks = self.detect_face_landmarks(image)
        if not landmarks: return None, None

        image_resized = cv2.resize(image, (224, 224))
        region_masks = self.generate_region_masks(image_resized.shape, landmarks, patch_size=16)

        # Normalize image
        image_tensor = np.transpose(image_resized.astype(np.float32) / 255.0, (2, 0, 1))
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image_normalized = (image_tensor - mean) / std

        features = {'image': image_normalized, 'region_masks': region_masks}
        return features, landmarks


class AverageLandmarkExtractor(FeatureExtractor):
    """Uses pre-computed average landmarks instead of detecting them."""

    def __init__(self, mean_landmarks):
        super().__init__()
        self.mean_landmarks = mean_landmarks

    def detect_face_landmarks(self, image):
        """Return average landmarks scaled to image size."""
        h, w = image.shape[:2]
        return [(idx, int(x_norm * w), int(y_norm * h)) for idx, x_norm, y_norm in self.mean_landmarks]


def process_single_image(args):
    """Worker function to process one image."""
    img_path, label, split, mean_landmarks = args
    try:
        image = cv2.imread(img_path)
        if image is None: return {'success': False, 'path': img_path, 'error': 'Read error'}

        # Use fallback if mean_landmarks are provided
        extractor = AverageLandmarkExtractor(mean_landmarks) if mean_landmarks else FeatureExtractor()
        augmentation = DataAugmentation() if split == 'train' and not mean_landmarks else None

        features, landmarks = extractor.extract_all_features(image, augmentation)

        if features is None:
            return {'success': False, 'path': img_path, 'error': 'No face detected'}

        h, w = image.shape[:2]
        normalized_landmarks = [(idx, x / w, y / h) for idx, x, y in landmarks] if landmarks else []

        return {'success': True, 'features': features, 'landmarks': normalized_landmarks, 'path': img_path,
                'label': label, 'split': split}
    except Exception as e:
        return {'success': False, 'path': img_path, 'error': str(e)}


def save_batch_to_file(results, output_file):
    """Save a batch of processed results to an HDF5 file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    mode = 'a' if os.path.exists(output_file) else 'w'
    with h5py.File(output_file, mode) as f:
        current_count = f.attrs.get('total_samples', 0)
        for i, result in enumerate(results):
            group = f.create_group(f"sample_{current_count + i}")
            for key, value in result['features'].items():
                group.create_dataset(key, data=value)
            group.attrs['path'] = result['path']
            group.attrs['label'] = result['label']
            group.attrs['split'] = result['split']
        f.attrs['total_samples'] = current_count + len(results)


def process_images_with_fallback(image_paths, labels, output_file, split, num_workers):
    """Process images, with a fallback mechanism for face detection failures."""
    print(f"Starting initial processing for {len(image_paths)} images ({split} set)...")
    tasks = [(path, label, split, None) for path, label in zip(image_paths, labels)]

    successful_results, failed_paths, landmark_collection = [], [], []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks),
                           desc=f"Processing {split}"):
            if result['success']:
                successful_results.append(result)
                if result['landmarks']: landmark_collection.append(result['landmarks'])
            else:
                failed_paths.append(result['path'])

    if not landmark_collection:
        print(f"Warning: No valid landmarks found in {split} set. Fallback may not be effective.")
        if successful_results: save_batch_to_file(successful_results, output_file)
        return

    # Calculate average landmarks
    avg_landmarks = {}
    for landmarks in landmark_collection:
        for idx, x, y in landmarks:
            if idx not in avg_landmarks: avg_landmarks[idx] = []
            avg_landmarks[idx].append((x, y))

    mean_landmarks = [(idx, np.mean([p[0] for p in points]), np.mean([p[1] for p in points])) for idx, points in
                      avg_landmarks.items()]

    if failed_paths:
        print(f"Processing {len(failed_paths)} failed images with average landmarks...")
        failed_tasks = [(path, labels[image_paths.index(path)], split, mean_landmarks) for path in failed_paths]
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap_unordered(process_single_image, failed_tasks), total=len(failed_tasks),
                               desc="Fallback"):
                if result['success']: successful_results.append(result)

    if successful_results:
        save_batch_to_file(successful_results, output_file)
        print(f"Saved {len(successful_results)} samples to {output_file}")
    gc.collect()


def process_dataset(dataset_dir, name, output_dir, is_real, num_workers):
    """Process an entire dataset, splitting into train/val/test."""
    print(f"\nProcessing dataset: {name}")
    image_paths = [os.path.join(r, f) for r, _, fs in os.walk(dataset_dir) for f in fs if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_paths: return
    random.shuffle(image_paths)

    n = len(image_paths)
    splits = {'train': image_paths[:int(0.6 * n)], 'val': image_paths[int(0.6 * n):int(0.8 * n)],
              'test': image_paths[int(0.8 * n):]}

    label = 0 if is_real else 1
    for split_name, paths in splits.items():
        if not paths: continue
        output_subdir = os.path.join(output_dir, "real" if is_real else "fake", name)
        os.makedirs(output_subdir, exist_ok=True)
        output_file = os.path.join(output_subdir, f"{split_name}_features.h5")
        process_images_with_fallback(paths, [label] * len(paths), output_file, split_name, num_workers)


def main():
    parser = argparse.ArgumentParser(description='Feature Extraction Script')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the datasets.')
    parser.add_argument('--output_dir', type=str, default='./features', help='Directory to save HDF5 feature files.')
    parser.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count() - 2),
                        help='Number of worker processes.')
    args = parser.parse_args()

    set_seed(42)

    real_dir = os.path.join(args.data_root, 'real')
    fake_dir = os.path.join(args.data_root, 'fake')

    for name in os.listdir(real_dir):
        process_dataset(os.path.join(real_dir, name), name, args.output_dir, is_real=True, num_workers=args.workers)

    for name in os.listdir(fake_dir):
        process_dataset(os.path.join(fake_dir, name), name, args.output_dir, is_real=False, num_workers=args.workers)


if __name__ == "__main__":
    main()