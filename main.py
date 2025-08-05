import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import gzip
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import openvino as ov
from openvino import serialize
import matplotlib.pyplot as plt

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
    """Prompt user to select CPU (0) or GPU (1) and return device."""
    choice = input("[INPUT] Select device for training (0=CPU, 1=GPU): ").strip()

    if choice == "1":
        if torch.cuda.is_available():
            print("[INFO] NVIDIA GPU detected: Using CUDA")
            return torch.device("cuda"), False, False
        elif has_ipex and hasattr(torch, "xpu") and torch.xpu.is_available():
            print("[INFO] Intel GPU detected: Using XPU (IPEX)")
            use_amp = input("[INPUT] Do you want to use Intel AMP (Automatic Mixed Precision)? (y/n): ").lower() == "y"
            return torch.device("xpu"), True, use_amp
        else:
            print("[WARN] No compatible GPU found (CUDA/IPEX unavailable). Fallback to CPU.")
            return torch.device("cpu"), False, False
    else:
        print("[INFO] Using CPU for training.")
        return torch.device("cpu"), False, False


# ===========================
# Dataset Loader
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
        try:
            return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx])
        except Exception as e:
            print(f"[WARN] Skipping invalid item at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.labels))


# ===========================
# CNN Model
# ===========================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# ===========================
# Training function with validation loss and plot
# ===========================
def train_model(model, train_loader, test_loader, device, epochs=1, intel_device=False, use_amp=False):
    print("[INFO] Starting training...")
    start_time = time.time()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if intel_device:
        import intel_extension_for_pytorch as ipex
        dtype = torch.bfloat16 if use_amp else None
        model, optimizer = ipex.optimize(
            model,
            optimizer=optimizer,
            dtype=dtype,
            inplace=True,
            fuse_update_step=True  # Use Optimizer Fusion
            # https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/technical_details/optimizer_fusion_gpu.html
        )

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"[TRAIN] Epoch [{epoch+1}/{epochs}]", unit="batch")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if intel_device and use_amp:
                with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                if intel_device and use_amp:
                    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        validation_losses.append(avg_val_loss)

        print(f"[INFO] Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"[INFO] Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    # Plot
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(training_losses, label="Training", color="blue")
    plt.plot(validation_losses, label="Validation", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    graph_path = results_dir / "loss_curve.png"
    plt.savefig(graph_path)
    print(f"[INFO] Saved loss graph to {graph_path}")

    return model



# ===========================
# Evaluation function (unchanged)
# ===========================
def evaluate_model(model, test_loader, device):
    print("[INFO] Starting evaluation...")
    model.eval()
    correct = 0

    try:
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()

        acc = correct / len(test_loader.dataset)
        print(f"[RESULT] Test Accuracy: {acc * 100:.2f}%")

    except RuntimeError as e:
        if "UR" in str(e) or "unregistered" in str(e).lower():
            print(f"[WARN] UR error during evaluation on {device}, switching to CPU fallback...")
            model.to("cpu")
            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images.to("cpu"))
                    pred = outputs.argmax(dim=1)
                    correct += (pred == labels).sum().item()
            acc = correct / len(test_loader.dataset)
            print(f"[RESULT] Test Accuracy (CPU fallback): {acc * 100:.2f}%")
        else:
            raise e



# ===========================
# Direct PyTorch → OpenVINO Conversion
# ===========================
def convert_to_openvino(model, input_shape, precision, output_dir="openvino_model"):
    print("[INFO] Starting PyTorch → OpenVINO conversion...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()

    current_device = next(model.parameters()).device
    if current_device.type != "cpu":
        print(f"[INFO] Moving model from {current_device} to CPU for conversion...")
        model = model.to("cpu")

    dummy_input = torch.randn(input_shape)

    import openvino as ov
    from openvino import serialize

    ov_model = ov.convert_model(model, example_input=dummy_input)

    precision_map = {"INT8": "int8", "BF16": "bf16", "FP16": "fp16", "FP32": "fp32"}
    p = precision_map.get(precision.upper(), "fp16")

    core = ov.Core()
    compiled_model = core.compile_model(ov_model, "CPU", config={"INFERENCE_PRECISION_HINT": p})

    model_path = output_path / f"model_{p}.xml"
    serialize(ov_model, str(model_path))
    print(f"[SUCCESS] Converted model saved at: {model_path}")

# ===========================
# Unzip MNIST Data if Needed
# ===========================
def unzip_mnist(data_path, gzipped_files, unzipped_files):
    print("[INFO] Unzipping MNIST data...")
    for gz_file, idx_file in zip(gzipped_files, unzipped_files):
        gz_path = data_path / gz_file
        idx_path = data_path / idx_file
        if gz_path.exists() and not idx_path.exists():
            with gzip.open(gz_path, 'rb') as f_in, open(idx_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"[INFO] Unzipped {gz_file} → {idx_file}")


# ===========================
# Main (Windows Guard)
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Recognition with OpenVINO 2025.2")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--precision", type=str,
                        choices=["INT8", "BF16", "FP16", "FP32"], help="Precision type for OpenVINO")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    args = parser.parse_args()

    # Interactive prompts if arguments not provided
    if args.batch_size is None:
        val = input("[INPUT] Enter batch size (default=128): ").strip()
        args.batch_size = int(val) if val else 128

    if args.precision is None:
        val = input("[INPUT] Enter precision (INT8/BF16/FP16/FP32) (default=FP16): ").strip().upper()
        args.precision = val if val in ["INT8", "BF16", "FP16", "FP32"] else "FP16"

    if args.epochs is None:
        val = input("[INPUT] Enter number of epochs (default=3): ").strip()
        args.epochs = int(val) if val else 3

    # ===========================
    # Check data availability
    # ===========================
    data_path = Path("data")
    unzipped_files = [
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
    ]
    gzipped_files = [f + ".gz" for f in unzipped_files]

    print("[INFO] Checking MNIST dataset availability...")
    if all((data_path / f).exists() for f in unzipped_files):
        print("[INFO] Using unzipped MNIST data files.")
    elif all((data_path / f).exists() for f in gzipped_files):
        unzip_mnist(data_path, gzipped_files, unzipped_files)
        print("[INFO] Using unzipped MNIST data files (after extraction).")
    else:
        print("[ERROR] No MNIST data found (neither .idx nor .gz files).")
        exit(1)

    # ===========================
    # Device Selection
    # ===========================
    device, intel_device, use_amp = get_device()

    # Show training variables
    print("\n========== Training Configuration ==========")
    print(f"Device      : {device}")
    print(f"Batch Size  : {args.batch_size}")
    print(f"Precision   : {args.precision}")
    print(f"Epochs      : {args.epochs}")
    print("============================================\n")

    # ===========================
    # Load dataset
    # ===========================
    print("[INFO] Loading MNIST dataset...")
    train_dataset = MNISTDataset(data_path / "train-images-idx3-ubyte",
                                 data_path / "train-labels-idx1-ubyte")
    test_dataset = MNISTDataset(data_path / "t10k-images-idx3-ubyte",
                                data_path / "t10k-labels-idx1-ubyte")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ===========================
    # Train & Evaluate
    # ===========================
    model = CNN()
    model = train_model(model, train_loader, test_loader, device, epochs=args.epochs,
                        intel_device=intel_device, use_amp=use_amp)
    evaluate_model(model, test_loader, device)

    # ===========================
    # Convert to OpenVINO
    # ===========================
    convert_to_openvino(model, (1, 1, 28, 28), args.precision)
