"""
Training Script for OncoVisionAI Multimodal Model
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_model import MultimodalCancerDetector, create_callbacks, F1Score
from src.data_preprocessing import CancerDataPreprocessor, MultimodalDataGenerator


def plot_training_history(history, save_path: str = 'outputs/training_history.png'):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history saved to {save_path}")


def evaluate_model(model, test_generator, save_path: str = 'outputs/evaluation_results.txt'):
    """Evaluate model on test set"""
    print("\n" + "="*80)
    print("EVALUATING MODEL ON TEST SET")
    print("="*80)
    
    # Convert test generator to arrays for Keras 3.x compatibility
    print("Preparing test data...")
    X_test_img = []
    X_test_clin = []
    y_test = []
    for i in range(len(test_generator)):
        batch = test_generator[i]
        X_test_img.append(batch[0][0])
        X_test_clin.append(batch[0][1])
        y_test.append(batch[1])
    
    X_test_img = np.concatenate(X_test_img, axis=0)
    X_test_clin = np.concatenate(X_test_clin, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    
    # Evaluate
    results = model.evaluate([X_test_img, X_test_clin], y_test, verbose=1)
    
    # Get metric names
    metric_names = model.metrics_names
    
    # Print results
    print("\nTest Results:")
    print("-" * 40)
    for name, value in zip(metric_names, results):
        print(f"{name.capitalize():20s}: {value:.4f}")
    print("-" * 40)
    
    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("OncoVisionAI - Test Set Evaluation\n")
        f.write("="*60 + "\n\n")
        for name, value in zip(metric_names, results):
            f.write(f"{name.capitalize():20s}: {value:.4f}\n")
    
    print(f"\n✓ Evaluation results saved to {save_path}")
    
    return dict(zip(metric_names, results))


def train_model(args):
    """Main training function"""
    print("\n" + "="*80)
    print("HEALTH SIGHT AI - TRAINING PIPELINE")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Image Directory: {args.image_dir}")
    print(f"  Clinical CSV: {args.clinical_csv}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Image Size: {args.img_size}x{args.img_size}")
    print("="*80 + "\n")
    
    # Prepare dataset
    print("Step 1: Preparing dataset...")
    preprocessor = CancerDataPreprocessor(img_size=(args.img_size, args.img_size))
    
    dataset = preprocessor.prepare_multimodal_dataset(
        image_dir=args.image_dir,
        clinical_csv=args.clinical_csv,
        test_size=0.2,
        val_size=0.1
    )
    
    # Create data generators
    print("\nStep 2: Creating data generators...")
    train_generator = MultimodalDataGenerator(
        image_paths=dataset['train']['image_paths'],
        clinical_features=dataset['train']['clinical'],
        labels=dataset['train']['labels'],
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        augment=True,
        shuffle=True
    )
    
    val_generator = MultimodalDataGenerator(
        image_paths=dataset['val']['image_paths'],
        clinical_features=dataset['val']['clinical'],
        labels=dataset['val']['labels'],
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        augment=False,
        shuffle=False
    )
    
    test_generator = MultimodalDataGenerator(
        image_paths=dataset['test']['image_paths'],
        clinical_features=dataset['test']['clinical'],
        labels=dataset['test']['labels'],
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        augment=False,
        shuffle=False
    )
    
    print(f"✓ Data generators created")
    print(f"  Training batches: {len(train_generator)}")
    print(f"  Validation batches: {len(val_generator)}")
    print(f"  Test batches: {len(test_generator)}")
    
    # Build model
    print("\nStep 3: Building multimodal model...")
    detector = MultimodalCancerDetector(
        img_size=(args.img_size, args.img_size, 3),
        num_clinical_features=5,
        num_classes=4,  # 4 classes: Normal, BCC, SCC, Melanoma
        dropout_rate=args.dropout
    )
    
    model = detector.build_fusion_model()
    detector.compile_model(learning_rate=args.learning_rate)
    detector.get_model_summary()
    
    # Create callbacks
    print("\nStep 4: Setting up callbacks...")
    callbacks = create_callbacks(
        model_checkpoint_path=args.checkpoint_path,
        tensorboard_log_dir=args.log_dir,
        early_stopping_patience=args.patience
    )
    
    # Train model
    print("\nStep 5: Training model...")
    print("="*80)
    
    # Use a simpler approach that works reliably with Keras 3.x
    print("Setting up training with manual epoch loop...")
    print("This ensures stable training without hanging issues.")
    
    # Manual training loop for better control
    from keras.callbacks import History
    history = History()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("="*80)
        
        # Training phase
        train_metrics = {'loss': [], 'accuracy': []}
        for batch_idx in range(len(train_generator)):
            batch = train_generator[batch_idx]
            if batch[0][0].shape[0] == 0:  # Skip empty batches
                continue
            
            # Train on batch
            metrics = model.train_on_batch(
                [batch[0][0], batch[0][1]],
                batch[1]
            )
            
            train_metrics['loss'].append(metrics[0])
            train_metrics['accuracy'].append(metrics[1])
            
            # Progress update every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = np.mean(train_metrics['loss'])
                avg_acc = np.mean(train_metrics['accuracy'])
                print(f"  Batch {batch_idx + 1}/{len(train_generator)} - "
                      f"loss: {avg_loss:.4f} - accuracy: {avg_acc:.4f}")
        
        # Calculate training metrics
        train_loss = np.mean(train_metrics['loss'])
        train_acc = np.mean(train_metrics['accuracy'])
        
        # Validation phase
        val_metrics = {'loss': [], 'accuracy': []}
        for batch_idx in range(len(val_generator)):
            batch = val_generator[batch_idx]
            if batch[0][0].shape[0] == 0:  # Skip empty batches
                continue
            
            # Evaluate on batch
            metrics = model.test_on_batch(
                [batch[0][0], batch[0][1]],
                batch[1]
            )
            
            val_metrics['loss'].append(metrics[0])
            val_metrics['accuracy'].append(metrics[1])
        
        # Calculate validation metrics
        val_loss = np.mean(val_metrics['loss'])
        val_acc = np.mean(val_metrics['accuracy'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} - Train Accuracy: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(args.checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        # Store history
        if not hasattr(history, 'history'):
            history.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        history.history['loss'].append(train_loss)
        history.history['accuracy'].append(train_acc)
        history.history['val_loss'].append(val_loss)
        history.history['val_accuracy'].append(val_acc)
    
    print("\n" + "="*80)
    print("✓ Training completed!")
    print("="*80)
    
    # Plot training history
    print("\nStep 6: Generating training plots...")
    plot_training_history(history, save_path=args.history_plot)
    
    # Evaluate on test set
    print("\nStep 7: Evaluating on test set...")
    test_results = evaluate_model(model, test_generator, save_path=args.eval_results)
    
    # Save final model
    print("\nStep 8: Saving final model...")
    final_model_path = args.final_model_path
    model.save(final_model_path)
    print(f"✓ Final model saved to {final_model_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Best Validation Loss: {min(history.history['val_loss']):.4f}")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    
    # Print test results (metric names may vary)
    if 'compile_metrics' in test_results:
        print(f"Test Accuracy: {test_results['compile_metrics']:.4f}")
    elif 'accuracy' in test_results:
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    
    if 'precision' in test_results and 'recall' in test_results:
        print(f"Test Precision: {test_results['precision']:.4f}")
        print(f"Test Recall: {test_results['recall']:.4f}")
        
        # Calculate F1 score
        precision = test_results['precision']
        recall = test_results['recall']
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        print(f"Test F1-Score: {f1:.4f}")
    
    print("="*80)
    
    print("\n✓ All done! Model is ready for deployment.")
    print(f"\nNext steps:")
    print(f"  1. Convert to TFLite: python models/export_tflite.py")
    print(f"  2. Generate Grad-CAM: python src/gradcam.py")
    print(f"  3. Run demo app: python app/demo_app.py")
    
    return model, history, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Health Sight AI Multimodal Model")
    
    # Data arguments
    parser.add_argument('--image-dir', type=str, default='data/raw/images',
                       help='Directory containing images')
    parser.add_argument('--clinical-csv', type=str, default='data/clinical_data.csv',
                       help='Path to clinical data CSV')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size (will be resized to img_size x img_size)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--checkpoint-path', type=str, 
                       default='models/checkpoints/best_model.keras',
                       help='Path to save best model checkpoint')
    parser.add_argument('--final-model-path', type=str,
                       default='models/saved_models/oncovision_multimodal.keras',
                       help='Path to save final trained model')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='TensorBoard log directory')
    parser.add_argument('--history-plot', type=str,
                       default='outputs/training_history.png',
                       help='Path to save training history plot')
    parser.add_argument('--eval-results', type=str,
                       default='outputs/evaluation_results.txt',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Train model
    train_model(args)
