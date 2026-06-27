# Metric Depth Estimation — Complete Project

## What This Project Does
Takes a single photo and predicts exactly how far away (in meters) every object is.

## How This Is Different From Your Earlier MiDaS Project (2D_3D Projection)
| Your Earlier Project (2D_3D) | THIS Project |
|------------------------------|-------------|
| Relative depth: 0.0 to 1.0 | Real depth: 2.3 meters, 5.1 meters, etc. |
| No camera info used | Uses camera focal length, sensor data |
| Pre-trained MiDaS only | You train your own model |
| Any random object | Train on new custom objects too |
| Visual comparison only | Accuracy metrics + focal length validation |

---

## COMPLETE SETUP GUIDE (Assuming Zero Knowledge)

### Step A: Install Required Software

1. Install VS Code — https://code.visualstudio.com/
2. Install Python 3.10+ — https://www.python.org/downloads/
   CRITICAL: Check "Add Python to PATH" during install
3. Install Git — https://git-scm.com/downloads/
4. (Optional) Install CUDA for NVIDIA GPU — https://developer.nvidia.com/cuda-downloads

### Step B: Open Project in VS Code

1. Open VS Code
2. File > Open Folder > select this metric_depth_project folder
3. Open Terminal: View > Terminal (or Ctrl+`)

### Step C: Create Virtual Environment and Install Packages

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## PROJECT EXECUTION ORDER

Step 1: Camera Calibration    → Find fx, fy, cx, cy
Step 2: Get Training Data     → NYU Depth V2 or your own photos
Step 3: Train the Model       → Custom depth estimation network
Step 4: Predict Depth         → Run on any new image
Step 5: Validate Accuracy     → Check focal length + metrics

See each step's folder for detailed instructions.
