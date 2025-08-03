# MNIST-Recognition-with-PyTorch-OpenVINO
This project demonstrates how to train a Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset using PyTorch and convert it directly into OpenVINO IR format for optimized inference.
It includes training, evaluation, visualization, and testing with multiple precision formats (INT8, BF16, FP16, FP32).

Features
- CNN-based classification for MNIST digits (28×28 grayscale images).  
- Automatic data preparation (unzips .gz MNIST files if needed).    
- Support for CPU, NVIDIA GPU (CUDA), and Intel GPU (XPU with IPEX).  
- Mixed-precision support: INT8, BF16, FP16, FP32.  
- Direct PyTorch → OpenVINO IR conversion (no ONNX required).  
- Progress display with tqdm.  
- Validation loss tracked each epoch with loss curve saved as results/loss_curve.png.

Evaluation outputs:
- Accuracy
- Test time
- Confusion matrix (results/confusion_matrix.png)
- Sample predictions (results/sample_predictions.png)
- Compatible with Windows and Linux.

Requirements   
1. Python dependencies   
Please install: torch torchvision intel_extension_for_pytorch openvino tqdm matplotlib scikit-learn

3. Hardware dependencies   
Intel GPU users: Install Intel® oneAPI runtime & drivers.   
NVIDIA GPU users: Install CUDA Toolkit and compatible drivers.   
CPU: No extra setup needed.

Usage
1. Prepare MNIST data   
Download the 4 MNIST .gz files and place them in the data/ directory.   
The program will automatically unzip them into .idx if needed.

2. Train and Convert Model
Run the training script:
python main.py

Prompts:   
    Batch size (default: 128)   
    Precision (INT8/BF16/FP16/FP32; default: FP16)     
    Number of epochs (default: 3)

Device selection (CPU or GPU)   
If Intel GPU selected → option to enable Intel AMP   

Saves:   
Model converted to OpenVINO IR in openvino_model/   
Loss curve: results/loss_curve.png  

3. Test Model   
Evaluate trained OpenVINO model:   
python test_model.py
 
Prompts:   
    Device selection (CPU/GPU)   
    Model precision (default: FP16)   

Saves and displays:   
Sample predictions (results/sample_predictions.png)    
Confusion matrix (results/confusion_matrix.png)

Example Output:    
[RESULT] Test Accuracy: 98.73%   
[INFO] Total test time: 3.55 seconds   
[INFO] Saved sample predictions to results/sample_predictions.png   
[INFO] Saved confusion matrix to results/confusion_matrix.png   
Results   
1. Loss Curve   
2. Sample Predictions   
3. Confusion Matrix

Supported Devices   
CPU: Default fallback.   
NVIDIA GPU (CUDA): Automatically used if available.   
Intel GPU (XPU with IPEX): Requires Intel® Extension for PyTorch and oneAPI drivers.

