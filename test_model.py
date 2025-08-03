import openvino as ov
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gzip
import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===========================
# Try Intel Extension for PyTorch (IPEX)
# ===========================
try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False


# ===========================
# Device Selection
# ===========================
def get_device():
    """Ask the user for CPU (0) or GPU (1) and return the device."""
    choice = input("[INPUT] Select device for testing (0=CPU, 1=GPU): ").strip()

    if choice == "1":
        if torch.cuda.is_available():
            print("[INFO] NVIDIA GPU detected: Using CUDA")
            return torch.device("cuda"), False
        elif has_ipex and hasattr(torch, "xpu") and torch.xpu.is_available():
            print("[INFO] Intel GPU detected: Using XPU (IPEX)")
            return torch.device("xpu"), True
        else:
            print("[WARN] No compatible GPU found (CUDA/IPEX unavailable). Fallback to CPU.")
            return torch.device("cpu"), False
    else:
        print("[INFO] Using CPU for testing.")
        return torch.device("cpu"), False


# ===========================
# Unzip MNIST Data if Needed
# ===========================
def unzip_mnist(data_path, gzipped_files, unzipped_files):
    print("[INFO] Unzipping MNIST test data...")
    for gz_file, idx_file in zip(gzipped_files, unzipped_files):
        gz_path = data_path / gz_file
        idx_path = data_path / idx_file
        if gz_path.exists() and not idx_path.exists():
            with gzip.open(gz_path, 'rb') as f_in, open(idx_path, 'wb') as f_out:
                f_out.write(f_in.read())
            print(f"[INFO] Unzipped {gz_file} → {idx_file}")


# ===========================
# MNIST Dataset Loader
# ===========================
class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = self._load_images(images_path)
        self.labels = self._load_labels(labels_path)

    def _load_images(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Missing file: {path}")
        with open(path, 'rb') as f:
            data = f.read()
        magic, num, rows, cols = np.frombuffer(data[:16], dtype='>i4')
        images = np.frombuffer(data[16:], dtype=np.uint8).reshape(num, 1, rows, cols)
        return images.astype(np.float32) / 255.0

    def _load_labels(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Missing file: {path}")
        with open(path, 'rb') as f:
            data = f.read()
        magic, num = np.frombuffer(data[:8], dtype='>i4')
        return np.frombuffer(data[8:], dtype=np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ===========================
# Check MNIST test data availability
# ===========================
data_path = Path("data")
unzipped_files = ["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
gzipped_files = [f + ".gz" for f in unzipped_files]

print("[INFO] Checking MNIST test dataset availability...")
if all((data_path / f).exists() for f in unzipped_files):
    print("[INFO] Using unzipped MNIST test data files.")
elif all((data_path / f).exists() for f in gzipped_files):
    unzip_mnist(data_path, gzipped_files, unzipped_files)
    print("[INFO] Using unzipped MNIST test data files (after extraction).")
else:
    print("[ERROR] No MNIST test data found (neither .idx nor .gz files).")
    exit(1)


# ===========================
# Select device for testing
# ===========================
device, intel_device = get_device()

# ===========================
# Load model based on user precision
# ===========================
precision = input("[INPUT] Select model precision (INT8/BF16/FP16/FP32) (default=FP16): ").strip().upper()
if precision not in ["INT8", "BF16", "FP16", "FP32"]:
    precision = "FP16"

model_dir = Path("openvino_model")
xml_path = model_dir / f"model_{precision.lower()}.xml"
bin_path = model_dir / f"model_{precision.lower()}.bin"

if not xml_path.exists() or not bin_path.exists():
    print(f"[ERROR] Model files for {precision} not found! ({xml_path}, {bin_path})")
    exit(1)

print(f"[INFO] Loading model: {xml_path}")
core = ov.Core()
compiled_model = core.compile_model(xml_path, "CPU")  # OpenVINO models run on CPU

# ===========================
# Load test dataset
# ===========================
test_images_path = data_path / "t10k-images-idx3-ubyte"
test_labels_path = data_path / "t10k-labels-idx1-ubyte"
test_dataset = MNISTDataset(test_images_path, test_labels_path)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ===========================
# Evaluate model with tqdm and timer
# ===========================
print("[INFO] Starting evaluation...")
start_time = time.time()

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

correct = 0
total = 0
all_preds = []
all_labels = []

for images, labels in tqdm(test_loader, desc="[TEST]", unit="batch"):
    images = np.array(images, dtype=np.float32)
    labels = labels.numpy()  # Convert to NumPy

    results = compiled_model([images])[output_layer]
    preds = np.argmax(results, axis=1)

    correct += np.sum(preds == labels)
    total += len(labels)

    all_preds.extend(preds)
    all_labels.extend(labels)

accuracy = correct / total * 100
end_time = time.time()
test_time = end_time - start_time

print(f"[RESULT] Test Accuracy: {accuracy:.2f}%")
print(f"[INFO] Total test time: {test_time:.2f} seconds")

# ===========================
# Ensure results directory exists
# ===========================
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

# ===========================
# Plot and save sample predictions
# ===========================
def plot_samples(images, labels, preds, n=10):
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i][0], cmap="gray")
        plt.title(f"T:{labels[i]}\nP:{preds[i]}",
                  color="green" if labels[i] == preds[i] else "red")
        plt.axis("off")
    plt.suptitle("MNIST Predictions (Samples)")
    sample_path = results_dir / "sample_predictions.png"
    plt.savefig(sample_path)
    print(f"[INFO] Saved sample predictions to {sample_path}")
    plt.show()

sample_images, sample_labels = next(iter(test_loader))
sample_preds = compiled_model([np.array(sample_images, dtype=np.float32)])[output_layer]
sample_preds = np.argmax(sample_preds, axis=1)
plot_samples(sample_images, sample_labels.numpy(), sample_preds)

# ===========================
# Plot and save confusion matrix
# ===========================
print("[INFO] Generating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix - MNIST Test Data")

cm_path = results_dir / "confusion_matrix.png"
plt.savefig(cm_path)
print(f"[INFO] Saved confusion matrix to {cm_path}")
plt.show()
