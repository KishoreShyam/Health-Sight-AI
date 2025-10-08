@echo off
echo ================================================================================
echo HEALTH SIGHT AI - ISIC DATASET TRAINING
echo Achieve 98%% Accuracy with Real Medical Data
echo ================================================================================
echo.

echo Step 1: Preparing ISIC Dataset...
echo ================================================================================
python prepare_isic_dataset.py
if errorlevel 1 (
    echo.
    echo ERROR: Dataset preparation failed!
    echo Please check that your ISIC files are in data\isic\
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Step 2: Starting Model Training...
echo ================================================================================
echo.
echo Training Configuration:
echo   - Dataset: ISIC (Real Medical Images)
echo   - Epochs: 150
echo   - Batch Size: 32
echo   - Expected Time: 24-36 hours
echo   - Expected Accuracy: 98%%+
echo.
echo Press Ctrl+C to cancel, or
pause

python models/train.py ^
    --image-dir data/isic_prepared/images ^
    --clinical-csv data/isic_prepared/prepared_metadata.csv ^
    --epochs 150 ^
    --batch-size 32 ^
    --learning-rate 0.00003 ^
    --img-size 224 ^
    --dropout 0.4 ^
    --patience 20

if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Training Complete!
echo ================================================================================
echo.
echo Your model is ready at: models/saved_models/oncovision_multimodal.keras
echo.
echo Next Steps:
echo   1. Check results: outputs/training_history.png
echo   2. Launch demo: streamlit run app/demo_app.py
echo.
pause
