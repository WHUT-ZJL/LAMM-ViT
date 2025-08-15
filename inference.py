import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score
import numpy as np
import random
import warnings

# Import the model architecture from model.py
from model import MaskGuidedVisionTransformer

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MaskedFeaturesDataset(Dataset):
    """Loads pre-extracted features from HDF5 files."""

    def __init__(self, feature_files):
        self.sample_info = []
        for file in feature_files if isinstance(feature_files, list) else [feature_files]:
            if os.path.exists(file):
                with h5py.File(file, 'r') as f:
                    self.sample_info.extend([(file, name) for name in f if name.startswith('sample_')])

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        file, name = self.sample_info[idx]
        with h5py.File(file, 'r') as f:
            sample = f[name]
            image = torch.tensor(sample['image'][:], dtype=torch.float32)
            masks = torch.tensor(sample['region_masks'][:], dtype=torch.float32)
            label = torch.tensor(sample.attrs['label'], dtype=torch.long)
            return image, masks, label


class MixedDataset(Dataset):
    """Mixes one fake dataset with real datasets for balanced evaluation."""

    def __init__(self, fake_dataset, real_datasets):
        total_real_samples = sum(len(d) for d in real_datasets)
        num_samples_per_class = min(len(fake_dataset), total_real_samples)

        self.fake_indices = random.sample(range(len(fake_dataset)), num_samples_per_class)

        all_real_samples = []
        for d in real_datasets:
            all_real_samples.extend([(d, i) for i in range(len(d))])
        self.real_samples = random.sample(all_real_samples, num_samples_per_class)

        self.fake_dataset = fake_dataset
        self.samples_per_class = num_samples_per_class

    def __len__(self):
        return self.samples_per_class * 2

    def __getitem__(self, idx):
        if idx < self.samples_per_class:
            # Fake sample, label 1
            img, mask, _ = self.fake_dataset[self.fake_indices[idx]]
            return img, mask, torch.tensor(1, dtype=torch.long)
        else:
            # Real sample, label 0
            dataset, sample_idx = self.real_samples[idx - self.samples_per_class]
            img, mask, _ = dataset[sample_idx]
            return img, mask, torch.tensor(0, dtype=torch.long)


def evaluate_dataset(model, dataloader, device):
    """Evaluate the model on a given dataloader."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for image, masks, labels in tqdm(dataloader, desc="Evaluating"):
            image, masks = image.to(device), masks.to(device)
            outputs = model(image, masks)
            probs = F.softmax(outputs['logits'], dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.argmax(all_probs, axis=1)

    acc = accuracy_score(all_labels, all_preds) * 100
    ap = average_precision_score(all_labels, all_probs[:, 1])

    return {'accuracy': acc, 'ap': ap}


def main():
    set_seed(42)

    # --- Configuration ---
    model_path = './output_model/best_model.pth'
    feature_dir = './features'
    batch_size = 64
    num_workers = 16

    # Real datasets (negative class)
    real_data_paths = [os.path.join(feature_dir, 'real', name, 'test_features.h5') for name in
                       os.listdir(os.path.join(feature_dir, 'real'))]

    # All fake datasets to test (positive class)
    fake_data_dir = os.path.join(feature_dir, 'fake')
    fake_files = [os.path.join(fake_data_dir, name, 'test_features.h5') for name in os.listdir(fake_data_dir)]

    # --- Initialization ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_params = {'num_classes': 2, 'dim': 768, 'num_layers': 12, 'num_heads': 12, 'mlp_dim': 3072}
    model = MaskGuidedVisionTransformer(**model_params).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    real_datasets = [MaskedFeaturesDataset(p) for p in real_data_paths if os.path.exists(p)]
    if not real_datasets:
        print("Error: No real datasets found for evaluation.")
        return

    # --- Evaluation Loop ---
    all_results = {}
    acc_values, ap_values = [], []

    for fake_file in fake_files:
        if not os.path.exists(fake_file): continue

        dataset_name = os.path.basename(os.path.dirname(fake_file))
        print(f"\n--- Evaluating dataset: {dataset_name} ---")

        fake_dataset = MaskedFeaturesDataset(fake_file)
        if len(fake_dataset) == 0:
            print("Dataset is empty, skipping.")
            continue

        mixed_dataset = MixedDataset(fake_dataset, real_datasets)
        mixed_loader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        metrics = evaluate_dataset(model, mixed_loader, device)
        all_results[dataset_name] = metrics

        acc_values.append(metrics['accuracy'])
        ap_values.append(metrics['ap'])

        print(f"Results -> ACC: {metrics['accuracy']:.2f}%, AP: {metrics['ap']:.4f}")

    # --- Final Summary ---
    mean_acc = np.mean(acc_values) if acc_values else 0
    mean_ap = np.mean(ap_values) if ap_values else 0

    all_results["Mean"] = {"accuracy": mean_acc, "ap": mean_ap}

    print("\n" + "=" * 30)
    print(f"Overall Mean ACC: {mean_acc:.2f}%")
    print(f"Overall Mean AP: {mean_ap:.4f}")
    print("=" * 30)

    with open('prediction_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\nEvaluation complete. Results saved to prediction_results.json")


if __name__ == "__main__":
    main()