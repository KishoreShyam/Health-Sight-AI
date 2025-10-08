# 🚀 OncoVisionAI - Setup Guide

## Quick Setup Instructions

### Step 1: Install Python (if not already installed)

1. Download Python 3.8+ from: https://www.python.org/downloads/
2. During installation, **check "Add Python to PATH"**
3. Verify installation:
   ```bash
   python --version
   ```

### Step 2: Install Dependencies

Open PowerShell or Command Prompt in the project directory and run:

```bash
# Navigate to project directory
cd "e:\CMR Hackathon"

# Install all dependencies
python -m pip install -r requirements.txt
```

**OR** install key packages individually:

```bash
python -m pip install streamlit plotly tensorflow opencv-python pillow numpy pandas matplotlib
```

### Step 3: Generate Sample Data (Quick Test)

```bash
python src/data_preprocessing.py --prepare-all --num-samples 200
```

This creates sample images and clinical data for testing.

### Step 4: Launch the Beautiful App

```bash
streamlit run app/demo_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## Alternative: Using Virtual Environment (Recommended)

### Create Virtual Environment

```bash
# Navigate to project
cd "e:\CMR Hackathon"

# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate again
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Launch App with Virtual Environment

```bash
# Make sure virtual environment is activated (you'll see (venv) in prompt)
streamlit run app/demo_app.py
```

---

## Troubleshooting

### Issue: "python is not recognized"

**Solution**: Add Python to PATH
1. Search for "Environment Variables" in Windows
2. Edit "Path" variable
3. Add Python installation directory (e.g., `C:\Python39\`)
4. Restart terminal

### Issue: "pip is not recognized"

**Solution**: Use `python -m pip` instead of `pip`
```bash
python -m pip install streamlit
```

### Issue: "Execution policy error" (PowerShell)

**Solution**: 
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Dependencies fail to install

**Solution**: Install one by one
```bash
python -m pip install streamlit
python -m pip install plotly
python -m pip install tensorflow
python -m pip install opencv-python
python -m pip install pillow
```

### Issue: "No module named 'streamlit'"

**Solution**: Ensure you're in the correct environment
```bash
# Check Python location
where python

# Install streamlit
python -m pip install streamlit

# Verify installation
python -c "import streamlit; print(streamlit.__version__)"
```

---

## Quick Demo (Without Training)

If you just want to see the app interface without a trained model:

1. Install dependencies:
   ```bash
   python -m pip install streamlit plotly pillow numpy
   ```

2. Launch app:
   ```bash
   streamlit run app/demo_app.py
   ```

3. The app will show an error about missing model, but you can see the beautiful UI!

---

## Full Pipeline (With Training)

For complete functionality with trained model:

```bash
# 1. Install all dependencies
python -m pip install -r requirements.txt

# 2. Run complete pipeline (this will take 20-30 minutes)
python run_pipeline.py --num-samples 1000 --epochs 20

# 3. Launch app
streamlit run app/demo_app.py
```

---

## Minimal Installation (Just to See UI)

If you want to quickly see the beautiful app interface:

```bash
pip install streamlit plotly pillow numpy pandas
streamlit run app/demo_app.py
```

You can explore the UI, tabs, and design even without a trained model!

---

## System Requirements

### Minimum
- Windows 10 or later
- Python 3.8+
- 4 GB RAM
- 2 GB free disk space

### Recommended
- Windows 11
- Python 3.9+
- 8 GB RAM
- 5 GB free disk space
- GPU (for faster training)

---

## Next Steps After Setup

1. ✅ Install Python and dependencies
2. ✅ Generate sample data (optional)
3. ✅ Launch the app
4. ✅ Explore the beautiful UI
5. ✅ Review documentation files
6. ✅ Practice your demo
7. ✅ Win the hackathon! 🏆

---

## Quick Commands Reference

```bash
# Check Python version
python --version

# Install dependencies
python -m pip install -r requirements.txt

# Generate sample data
python src/data_preprocessing.py --prepare-all --num-samples 200

# Train model (quick)
python models/train.py --epochs 5 --batch-size 32

# Launch app
streamlit run app/demo_app.py

# Run complete pipeline
python run_pipeline.py
```

---

## Need Help?

1. Check **FAQ.md** for common questions
2. Review **QUICKSTART.md** for detailed instructions
3. See **FINAL_SUMMARY.md** for complete overview

---

**OncoVisionAI** - *Ready to launch!* 🚀
