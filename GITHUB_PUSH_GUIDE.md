# GitHub Push Guide - Health Sight AI

## Quick Push Commands

### Step 1: Initialize Git (if not already done)
```bash
cd "E:\CMR Hackathon"
git init
```

### Step 2: Add Remote Repository
```bash
git remote add origin https://github.com/KishoreShyam/Health-Sight-AI.git
```

### Step 3: Add All Files
```bash
git add .
```

### Step 4: Commit
```bash
git commit -m "Initial commit: Health Sight AI - Multimodal Cancer Detection System"
```

### Step 5: Push to GitHub
```bash
git branch -M main
git push -u origin main
```

---

## If Repository Already Exists

If you get an error that the repository already has content:

### Option A: Force Push (Overwrites remote)
```bash
git push -u origin main --force
```

### Option B: Pull First, Then Push
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## Complete Step-by-Step Guide

### 1. Open PowerShell/Terminal
```powershell
cd "E:\CMR Hackathon"
```

### 2. Check Git Status
```bash
git status
```

### 3. Configure Git (First Time Only)
```bash
git config --global user.name "KishoreShyam"
git config --global user.email "your-email@example.com"
```

### 4. Initialize Repository
```bash
# If not already initialized
git init

# Add remote
git remote add origin https://github.com/KishoreShyam/Health-Sight-AI.git

# Or if remote already exists, update it
git remote set-url origin https://github.com/KishoreShyam/Health-Sight-AI.git
```

### 5. Stage All Files
```bash
git add .
```

### 6. Check What Will Be Committed
```bash
git status
```

You should see:
- ✅ All Python files (.py)
- ✅ Documentation (.md)
- ✅ Configuration files
- ❌ Large data files (excluded by .gitignore)
- ❌ Model files (excluded by .gitignore)

### 7. Commit Changes
```bash
git commit -m "Initial commit: Health Sight AI - Multimodal Cancer Detection System

Features:
- Multimodal fusion architecture (CNN + MLP)
- Transfer learning with EfficientNet/MobileNet
- Advanced augmentation (Mixup, strong transforms)
- Streamlit demo application
- Training scripts for various scenarios
- Comprehensive documentation"
```

### 8. Push to GitHub
```bash
git branch -M main
git push -u origin main
```

---

## What Gets Pushed (and What Doesn't)

### ✅ WILL BE PUSHED:
- All Python source code (.py)
- Documentation files (.md)
- Configuration files (requirements.txt, etc.)
- Project structure (folders)
- README and guides
- Small assets (if any)

### ❌ WON'T BE PUSHED (Excluded by .gitignore):
- Large dataset files (data/isic_prepared/)
- Trained model files (.keras, .h5)
- Training logs (logs/)
- Checkpoints (models/checkpoints/)
- CSV data files
- Image files (.jpg, .png)
- Python cache (__pycache__)
- Virtual environments

---

## Troubleshooting

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/KishoreShyam/Health-Sight-AI.git
```

### Error: "failed to push some refs"
```bash
# Pull first
git pull origin main --allow-unrelated-histories

# Then push
git push -u origin main
```

### Error: "Authentication failed"
You need to use a Personal Access Token (PAT) instead of password:

1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo` (all)
4. Copy the token
5. Use it as password when pushing

Or use SSH:
```bash
git remote set-url origin git@github.com:KishoreShyam/Health-Sight-AI.git
```

### Error: "Repository too large"
This means some large files are being pushed. Check:
```bash
git ls-files --cached | xargs ls -lh | sort -k5 -hr | head -20
```

Remove large files:
```bash
git rm --cached path/to/large/file
git commit -m "Remove large files"
```

---

## After Pushing

### Verify on GitHub
1. Go to: https://github.com/KishoreShyam/Health-Sight-AI
2. Check that all files are there
3. Verify README.md displays correctly

### Add Repository Description
On GitHub repository page:
- Click "⚙️ Settings"
- Add description: "AI-powered multimodal cancer detection system for early screening in rural communities"
- Add topics: `machine-learning`, `healthcare`, `cancer-detection`, `deep-learning`, `tensorflow`, `keras`, `streamlit`

### Create Releases (Optional)
```bash
git tag -a v1.0.0 -m "Initial release: Health Sight AI v1.0.0"
git push origin v1.0.0
```

---

## Future Updates

### To Push New Changes:
```bash
# Stage changes
git add .

# Commit
git commit -m "Description of changes"

# Push
git push origin main
```

### To Pull Latest Changes:
```bash
git pull origin main
```

---

## Repository Structure on GitHub

```
Health-Sight-AI/
├── README.md                          ✅ Pushed
├── requirements.txt                   ✅ Pushed
├── .gitignore                         ✅ Pushed
├── app/
│   ├── demo_app.py                   ✅ Pushed
│   └── assets/                       ✅ Pushed (if small)
├── models/
│   ├── multimodal_model.py           ✅ Pushed
│   ├── saved_models/                 ❌ Not pushed (too large)
│   └── checkpoints/                  ❌ Not pushed (too large)
├── src/
│   ├── data_preprocessing.py         ✅ Pushed
│   └── ...                           ✅ Pushed
├── train_improved_model.py           ✅ Pushed
├── train_efficient_small_data.py     ✅ Pushed
├── prepare_my_isic_data.py           ✅ Pushed
├── data/
│   └── isic_prepared/                ❌ Not pushed (too large)
└── docs/                             ✅ Pushed
```

---

## Important Notes

### 1. Large Files
GitHub has a 100MB file size limit. Your dataset and models are excluded automatically.

### 2. Sensitive Information
Make sure no API keys, passwords, or sensitive data are in the code.

### 3. Model Sharing
To share trained models, use:
- **Google Drive** - Share link in README
- **Hugging Face** - Upload to model hub
- **GitHub Releases** - For smaller models (<100MB)

### 4. Dataset Instructions
Add to README:
```markdown
## Dataset Setup
1. Download ISIC 2019 dataset from [link]
2. Run: `python prepare_my_isic_data.py`
3. Train: `python train_efficient_small_data.py`
```

---

## Quick Reference

```bash
# Complete push sequence
cd "E:\CMR Hackathon"
git init
git remote add origin https://github.com/KishoreShyam/Health-Sight-AI.git
git add .
git commit -m "Initial commit: Health Sight AI"
git branch -M main
git push -u origin main
```

---

## Need Help?

If you encounter issues:
1. Check `.gitignore` is working: `git status`
2. Verify remote URL: `git remote -v`
3. Check branch: `git branch`
4. View commit history: `git log --oneline`

---

**Ready to push!** Just run the commands in order. 🚀
