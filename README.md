# S²UG-Mamba Inference Tool

This repository provides the executable inference code and compiled modules for （Spatio-Spectral Uncertainty-Gated Mamba Network ,S²UG-Mamba）. 

> **⚠️ IMPORTANT SYSTEM REQUIREMENTS:**
> To protect the core intellectual property (e.g., the Frequency Module and Cross-Scan Mamba) during the review process, the core network architectures have been compiled into C-extensions (`.so` files). 
> **Therefore, this package MUST be run on a Linux (x86_64) machine with Python 3.10.**

---

## 1. Directory Structure

Please ensure your unzipped directory is structured exactly like this:

```text
SeGMaNet_Release/
 ├── inference.py                 # Main inference script (Entry point)
 ├── requirements.txt             # Basic pip dependencies
 ├── weight_xxxx.pt               # Pre-trained model weights (Add your weights here)
 ├── config.cpython-310...so      # Compiled dependency
 ├── engine_train.cpython-310...so# Compiled dependency
 ├── loss.cpython-310...so        # Compiled dependency
 ├── train.cpython-310...so       # Compiled dependency
 ├── utils.cpython-310...so       # Compiled dependency
 └── models/
      ├── __init__.py             # Python package indicator
      ├── frequency_module6.cpython-310...so 
      ├── mamba_x.cpython-310...so           
      └── models.cpython-310...so            
```

---

## 2. Environment Setup

We strongly recommend using **Conda** to create an isolated, clean environment to avoid dependency conflicts.

### Step 2.1: Create & Activate Environment
```bash
conda create -n segmanet python=3.10 -y
conda activate ssugnet
```

### Step 2.2: Install Basic Dependencies
```bash
pip install -r requirements.txt
```

### Step 2.3: Install Mamba (Crucial for Full Model)
If you intend to test the full SeGMaNet with Mamba enhancement modules, you must install the `mamba-ssm` package. Please ensure your PyTorch and CUDA versions are strictly compatible.
```bash
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```
*(Note: If Mamba installation fails due to local hardware limitations, you can still run the model by appending the `--disable_mamba` flag during inference, provided you use the corresponding non-Mamba ablation weights).*

---

## 3. How to Run Inference

Use the `inference.py` script to generate Saliency Maps from your input images. The script utilizes `argparse` for flexible path configurations.

### Standard Command
```bash
python inference.py \
    --img_dir "/path/to/your/input_images" \
    --weight_path "/path/to/your/weight_file.pt" \
    --output_dir "./outputs"
```

### Optional Arguments
* `--backbone`: Set the backbone network. (Default: `convnext_tiny`).
* `--disable_mamba`: Add this flag if you are testing an ablation weight that does not include Mamba modules.
* `--img_size`: Input resolution for the model. (Default: `320`).

### Example (Ablation testing without Mamba)
```bash
python inference.py \
    --img_dir "./demo_images" \
    --weight_path "./weights/ablation_no_mamba.pt" \
    --output_dir "./results" \
    --disable_mamba
```

---

## 4. Notes on Output & Metrics (KL Divergence)(maybe you need？maybe not)

The inference script automatically applies targeted post-processing to the raw network outputs:
1. **Gaussian Smoothing (`sigma=2.0`)**: Mitigates upsampling grid artifacts, making the spatial distribution closer to natural human visual attention.
2. **Floor Lifting (Anti-Dead-Zero)**: Maps the predicted probability range from `[0, 255]` to `[1, 255]`. This is specifically designed to prevent the "zero-denominator explosion" issue when calculating the **KL Divergence (KL)** metric using saved `.png` files.
