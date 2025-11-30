# Food-Detector Setup Guide

This document provides detailed setup instructions for running the Food-Detector project. There are two main use cases:
1. **Quick Demo** – Run the pre-trained model in `final-demo.ipynb`
2. **Full Development** – Train models, explore data, and run all components

---

## Prerequisites

- Python 3.9+ 
- macOS, Linux, or Windows (with bash or WSL)
- 4+ GB RAM (8+ GB recommended for model training)
- GPU support (optional, but recommended for YOLO training)

---

## Option 1: Quick Demo Setup (final-demo.ipynb)

If you only want to **run the pre-trained model and see the demo**, follow these quick steps.

### 1. Clone & Environment Setup

```bash
cd /path/to/your/workspace
git clone https://github.com/GaeaStorm/Food-Detector.git
cd Food-Detector
git checkout main  # or your working branch
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Minimal Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This installs all necessary libraries including:
- `ultralytics` (YOLO framework)
- `torch` & `torchvision` (PyTorch)
- `pandas`, `numpy`
- `jupyter`, `notebook` (for .ipynb files)
- `chromadb` (recipe database)

### 4. Verify Pre-trained Model

The demo expects the pre-trained model at:
```
runs/detect/train2/weights/best.pt
```

Check that this file exists:
```bash
ls -la runs/detect/train2/weights/best.pt
```

If missing, you have two options:
- **Option A**: Download from your shared drive or model artifacts
- **Option B**: Train a new model (see Option 2 below)

### 5. API Key (Optional but Recommended)

The demo uses OpenAI API to expand ingredient names and provide better recipe suggestions.

Create a `.env` file in the repo root:
```
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
```

Or set it in your shell:
```bash
export OPENAI_API_KEY="sk-proj-YOUR_KEY_HERE"
```

> **Without the API key**, the demo will still work but ingredient expansion and recipe suggestions may be less accurate.

### 6. Run the Demo

```bash
jupyter notebook final-demo.ipynb
```

Follow the cells in order:
1. **Cell 1**: Import dependencies
2. **Cell 2**: Load the pre-trained YOLO model
3. **Cell 3**: Detect ingredients in a test image
4. **Cell 4**: Expand ingredient list (with optional API)
5. **Cell 5**: Query the recipe database and suggest recipes

---

## Option 2: Full Development Setup (Training & Exploration)

If you want to **train models, explore the dataset, and run all components**, follow the complete setup.

### 1. Clone & Environment Setup

```bash
cd /path/to/your/workspace
git clone https://github.com/GaeaStorm/Food-Detector.git
cd Food-Detector
git checkout main  # or your working branch
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Full Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Setup Recipe Database (Optional)

If you want recipe suggestions, download the Recipe1M+ dataset:

```bash
bash setup.sh
```

This script downloads:
- `recipes_with_nutritional_info.json` (~200 MB)
- `det_ingrs.json` (ingredient embeddings)
- `recipe1M_layers.tar.gz` (full recipe layers)

> **Warning**: This may take 10-30 minutes depending on your internet connection.

Alternatively, if already downloaded, ensure:
```
dataset/recipe/
  ├── recipes_with_nutritional_info.json
  ├── det_ingrs.json
  └── layers/
```

### 5. Verify Dataset Structure

The Roboflow food ingredient dataset should be in:
```
dataset/
  ├── train/
  │   ├── images/
  │   ├── labels/
  │   └── _classes.csv
  ├── valid/
  │   ├── images/
  │   ├── labels/
  │   └── _classes.csv
  └── test/
      ├── images/
      ├── labels/
      └── _classes.csv
```

If not present:
1. Download from [Roboflow Universe](https://universe.roboflow.com/ai-project-pji0a/ingredient-detection-5uzov)
2. Extract to the `dataset/` folder
3. Ensure structure matches above

### 6. API Key

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-proj-YOUR_KEY_HERE"
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
```

### 7. Explore Data

Run the data exploration notebook:
```bash
jupyter notebook data_exploration.ipynb
```

This helps understand:
- Class distributions
- Image sizes
- Label formats
- Data quality issues

### 8. Train Models

**Option A: Train YOLO (Recommended)**

Use the `yolo_model.ipynb` notebook:
```bash
jupyter notebook yolo_model.ipynb
```

Or train from command line:
```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8l.pt')  # or yolov8m.pt, yolov8s.pt
results = model.train(data='dataset/data.yaml', epochs=100, imgsz=640)
"
```

**Option B: Train CNN (Alternative)**

Use the `trial_cnn.ipynb` notebook:
```bash
jupyter notebook trial_cnn.ipynb
```

> **Note**: YOLO training requires GPU support. For CPU-only machines, use smaller models or reduce epochs.

### 9. Export Trained Models

After training, export the best model:
```bash
cp runs/detect/train<N>/weights/best.pt ./my_model.pt
```

Update paths in notebooks to point to your new model.

### 10. Run Full Demo with Your Model

Update `final-demo.ipynb` to use your trained model:

```python
model = YOLO("runs/detect/train<N>/weights/best.pt")  # Replace <N> with your run number
```

Then run:
```bash
jupyter notebook final-demo.ipynb
```

---

## Common Files & Their Purpose

| File | Purpose | When Used |
|------|---------|-----------|
| `final-demo.ipynb` | End-to-end demo (detect → expand ingredients → suggest recipes) | Demo mode |
| `yolo_model.ipynb` | YOLO training & evaluation notebook | Model training |
| `trial_cnn.ipynb` | CNN training (alternative approach) | Model experimentation |
| `data_exploration.ipynb` | Data analysis and visualization | Understanding data |
| `helper.py` | Utility functions (detect_inventory, display_results, etc.) | Demo & notebooks |
| `query_recipes.py` | Recipe database query & suggestion logic | Demo & notebooks |
| `build_recipe_index.py` | Build ChromaDB recipe index | Setup (run once) |
| `final_demo.py` | Python script version of demo | Alternative to notebook |
| `requirements.txt` | Python dependencies | Setup |
| `setup.sh` | Download Recipe1M+ dataset | Full setup |

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'ultralytics'`
**Solution**: Install requirements again:
```bash
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: runs/detect/train2/weights/best.pt`
**Solution**: Either download the pre-trained model or train a new one (see Option 2, Step 8).

### Issue: CUDA out of memory during training
**Solution**: 
- Use a smaller model: `yolov8s.pt` instead of `yolov8l.pt`
- Reduce batch size in training config
- Use CPU-only: `device=cpu` (slower but works)

### Issue: `OpenAI API key not found`
**Solution**: The demo works without API key but ingredient expansion won't work. Set the key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Issue: Recipe database not loading
**Solution**: Ensure dataset/recipe/ exists and contains:
- `recipes_with_nutritional_info.json`
- `det_ingrs.json`

Run `bash setup.sh` or download manually from the sources listed in `setup.sh`.

### Issue: Jupyter kernel dies or crashes
**Solution**:
```bash
# Clear notebook cache
rm -rf .ipynb_checkpoints

# Restart kernel in notebook (top menu: Kernel > Restart & Clear Output)

# Or reinstall jupyter
pip uninstall jupyter notebook -y
pip install jupyter notebook
```

---

## Next Steps

1. **For Quick Start**: Follow **Option 1** (10 min setup)
2. **For Development**: Follow **Option 2** (30+ min setup depending on data downloads)
3. **For Training**: Use notebooks (`yolo_model.ipynb`, `trial_cnn.ipynb`)
4. **For Deployment**: Export trained model and integrate into a web app or API

---

## Project Structure Summary

```
Food-Detector/
├── README.md                      # Original README
├── README2.md                     # This setup guide
├── requirements.txt               # Python dependencies
├── setup.sh                       # Download Recipe1M+ dataset
│
├── final-demo.ipynb              # 🎯 Main demo (use this!)
├── final_demo.py                 # Python version of demo
├── helper.py                      # Demo utilities
├── query_recipes.py              # Recipe query logic
│
├── yolo_model.ipynb              # YOLO training
├── trial_cnn.ipynb               # CNN training
├── data_exploration.ipynb        # Data analysis
│
├── build_recipe_index.py         # Build recipe DB index
├── dataset/                       # Dataset folder (Roboflow data)
├── dataset/recipe/               # Recipe1M+ data (after setup.sh)
├── runs/                         # Training outputs (YOLO runs)
├── chroma_recipe_db/             # ChromaDB recipe database
└── .venv/                        # Virtual environment (created during setup)
```

---

## Questions or Issues?

- Check the **Troubleshooting** section above
- Review cell outputs in notebooks for detailed error messages
- Ensure all paths exist: `ls dataset/train/`, `ls runs/detect/train2/weights/best.pt`
- Verify virtual environment is activated: `which python` should show `.venv` path

---

**Last Updated**: November 2025  
**Version**: 1.0
