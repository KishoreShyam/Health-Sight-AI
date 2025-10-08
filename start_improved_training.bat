@echo off
echo ================================================================================
echo HEALTH SIGHT AI - IMPROVED TRAINING (ANTI-OVERFITTING)
echo ================================================================================
echo.
echo This script will train a new model with enhanced regularization to prevent
echo the overfitting issue you experienced.
echo.
echo Key Improvements:
echo   - Higher dropout rate (0.5)
echo   - Strong data augmentation
echo   - L2 regularization
echo   - Lower learning rate with decay
echo   - Class weights for imbalanced data
echo.
echo Expected Training Time: 6-10 hours
echo Expected Accuracy: 92-95%% (with better generalization)
echo.
echo ================================================================================
echo.

set /p confirm="Start training? (y/n): "
if /i not "%confirm%"=="y" (
    echo Training cancelled.
    pause
    exit /b
)

echo.
echo Starting training...
echo.

python train_improved_model.py --epochs 80 --batch-size 32 --learning-rate 0.0001 --dropout 0.5 --img-size 128

echo.
echo ================================================================================
echo Training Complete!
echo ================================================================================
echo.
echo Next steps:
echo   1. Check training plots: models\checkpoints\improved_training_history.png
echo   2. View TensorBoard: tensorboard --logdir logs\improved_training
echo   3. Run demo: streamlit run app\demo_app.py
echo.
pause
