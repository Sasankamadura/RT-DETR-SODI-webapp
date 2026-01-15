# Quick Start: Export and Run Locally

## Step 1: Export on Kaggle (Run This Now)

```python
# Copy-paste this in your Kaggle notebook
!python tools/export_onnx.py \
    -c /kaggle/working/RT-DETR_SOD_IMP/rtdetr_pytorch/configs/rtdetr/rtdetr_visdrone_r18vd.yml \
    -r /kaggle/input/rt-detr-r18-baseline-on-visdrone2019-c/checkpoint0093.pth \
    --check

# Download the generated rtdetr_r18.onnx file
```

## Step 2: On Your PC (Python 3.12)

### Install Dependencies
```powershell
pip install onnxruntime-gpu opencv-python pillow numpy
```

### Run Inference
```powershell
cd D:\IIT\RT-DETR\RT-DETR_SOD_IMP\rtdetr_pytorch

# Place rtdetr_r18.onnx in this directory

python local_inference.py --image "path\to\image.jpg"
```

**That's it!** No repo cloning, no PyTorch, just ONNX + image → detections! 🚀

## What You'll Have

```
rtdetr_pytorch\
├── rtdetr_r18.onnx           # ~90MB (from Kaggle)
└── local_inference.py         # Ready to use!
```

Total size: **~500MB** vs 5GB+ for full PyTorch setup!
