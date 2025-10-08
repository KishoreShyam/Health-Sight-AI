"""
Improved ISIC Training Script with Anti-Overfitting Techniques
Addresses overfitting issues with enhanced regularization and data augmentation
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.multimodal_model import MultimodalCancerDetector
from src.data_preprocessing import CancerDataPreprocessor, MultimodalDataGenerator
import keras
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def create_improved_data_generator(image_paths, clinical_features, labels, 
                                   batch_size=32, img_size=(128, 128), 
                                   augment=False, shuffle=True):
    """
    Enhanced data generator with stronger augmentation to prevent overfitting
    """
    try:
        from keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
        except ImportError:
            # Fallback: Use manual augmentation with cv2
            ImageDataGenerator = None
    from PIL import Image
    
    class ImprovedMultimodalGenerator(keras.utils.Sequence):
        def __init__(self, image_paths, clinical_features, labels, 
                     batch_size, img_size, augment, shuffle, **kwargs):
            super().__init__(**kwargs)
            self.image_paths = image_paths
            self.clinical_features = clinical_features
            self.labels = labels
            self.batch_size = batch_size
            self.img_size = img_size
            self.augment = augment
            self.shuffle = shuffle
            self.indices = np.arange(len(self.image_paths))
            
            # Enhanced augmentation for training
            if augment and ImageDataGenerator is not None:
                self.image_augmenter = ImageDataGenerator(
                    rotation_range=30,          # Increased rotation
                    width_shift_range=0.2,      # Increased shift
                    height_shift_range=0.2,
                    shear_range=0.2,            # Added shear
                    zoom_range=0.2,             # Increased zoom
                    horizontal_flip=True,
                    vertical_flip=True,         # Added vertical flip
                    brightness_range=[0.8, 1.2], # Added brightness variation
                    fill_mode='nearest'
                )
            else:
                self.image_augmenter = None
            
            self.on_epoch_end()
        
        def __len__(self):
            return int(np.ceil(len(self.image_paths) / self.batch_size))
        
        def _manual_augment(self, img_array):
            """Manual augmentation fallback using cv2 and numpy"""
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_array = np.fliplr(img_array)
            
            # Random vertical flip
            if np.random.rand() > 0.5:
                img_array = np.flipud(img_array)
            
            # Random rotation
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-30, 30)
                h, w = img_array.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                img_array = cv2.warpAffine(img_array, M, (w, h), 
                                          borderMode=cv2.BORDER_REFLECT)
            
            # Random brightness
            if np.random.rand() > 0.5:
                brightness_factor = np.random.uniform(0.8, 1.2)
                img_array = np.clip(img_array * brightness_factor, 0, 1)
            
            # Random zoom (crop and resize)
            if np.random.rand() > 0.5:
                zoom_factor = np.random.uniform(0.8, 1.0)
                h, w = img_array.shape[:2]
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                top = np.random.randint(0, h - new_h + 1) if new_h < h else 0
                left = np.random.randint(0, w - new_w + 1) if new_w < w else 0
                img_array = img_array[top:top+new_h, left:left+new_w]
                img_array = cv2.resize(img_array, (w, h))
            
            return img_array
        
        def __getitem__(self, index):
            indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
            
            batch_images = []
            batch_clinical = []
            batch_labels = []
            
            for idx in indices:
                try:
                    # Load image
                    img_path = self.image_paths[idx]
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0
                    
                    # Apply augmentation
                    if self.augment and self.image_augmenter:
                        img_array = self.image_augmenter.random_transform(img_array)
                    elif self.augment:
                        # Manual augmentation fallback
                        img_array = self._manual_augment(img_array)
                    
                    # Add Gaussian noise for regularization (only during training)
                    if self.augment:
                        noise = np.random.normal(0, 0.01, img_array.shape)
                        img_array = np.clip(img_array + noise, 0, 1)
                    
                    batch_images.append(img_array)
                    batch_clinical.append(self.clinical_features[idx])
                    batch_labels.append(self.labels[idx])
                    
                except Exception as e:
                    print(f"Error loading image {idx}: {e}")
                    continue
            
            if len(batch_images) == 0:
                return self.__getitem__((index + 1) % len(self))
            
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_clinical = np.array(batch_clinical, dtype=np.float32)
            batch_labels = keras.utils.to_categorical(batch_labels, num_classes=4)
            
            return (batch_images, batch_clinical), batch_labels
        
        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)
    
    return ImprovedMultimodalGenerator(
        image_paths, clinical_features, labels,
        batch_size, img_size, augment, shuffle
    )


def train_improved_model(epochs=80, batch_size=32, learning_rate=0.0001, 
                        dropout_rate=0.5, img_size=128, max_samples=None):
    """
    Train with improved anti-overfitting techniques:
    - Higher dropout rate
    - Stronger data augmentation
    - L2 regularization
    - Class weights for imbalanced data
    - Lower learning rate
    - Smaller image size for better generalization
    - Optional max_samples for faster testing
    """
    
    print("\n" + "="*80)
    print("HEALTH SIGHT AI - IMPROVED TRAINING (ANTI-OVERFITTING)")
    print("="*80)
    print(f"\nConfiguration:")
    if max_samples:
        print(f"  Dataset: ISIC Subset ({max_samples} images)")
    else:
        print(f"  Dataset: Full ISIC (25,331 images)")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate} (reduced)")
    print(f"  Dropout Rate: {dropout_rate} (increased)")
    print(f"  Image Size: {img_size}x{img_size} (smaller for generalization)")
    print(f"  Augmentation: STRONG (rotation, flip, zoom, brightness, noise)")
    print("="*80)
    
    # Load metadata
    print("\nStep 1: Loading ISIC metadata...")
    metadata_path = 'data/isic_prepared/prepared_metadata.csv'
    df = pd.read_csv(metadata_path)
    
    print(f"✓ Total samples: {len(df)}")
    print("\nClass distribution:")
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        class_name = df[df['label'] == label]['class_name'].iloc[0]
        pct = (count / len(df)) * 100
        print(f"  {class_name:12s}: {count:6d} ({pct:5.1f}%)")
    
    # Prepare dataset
    print("\nStep 2: Preparing dataset splits...")
    
    image_paths = []
    labels = []
    clinical_features = []
    
    for idx, row in df.iterrows():
        img_path = os.path.join('data/isic_prepared/images', row['image_id'])
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(row['label'])
            clinical_features.append([
                row['age'],
                row['symptom_duration_months'],
                row['family_history'],
                row['pain_score'],
                row['lesion_size_mm']
            ])
    
    labels = np.array(labels)
    clinical_features = np.array(clinical_features)
    
    print(f"✓ Loaded {len(image_paths)} valid image paths")
    
    # Limit samples if specified
    if max_samples and len(image_paths) > max_samples:
        print(f"\n⚠️  Limiting to {max_samples} samples for faster training...")
        indices = np.random.choice(len(image_paths), max_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]
        labels = labels[indices]
        clinical_features = clinical_features[indices]
        print(f"✓ Using {len(image_paths)} samples")
    
    # Split data with stratification
    X_train_paths, X_temp_paths, y_train, y_temp, X_train_clin, X_temp_clin = \
        train_test_split(image_paths, labels, clinical_features, 
                       test_size=0.3, random_state=42, stratify=labels)
    
    X_val_paths, X_test_paths, y_val, y_test, X_val_clin, X_test_clin = \
        train_test_split(X_temp_paths, y_temp, X_temp_clin,
                       test_size=0.5, random_state=42, stratify=y_temp)
    
    # Normalize clinical features
    scaler = StandardScaler()
    X_train_clin = scaler.fit_transform(X_train_clin)
    X_val_clin = scaler.transform(X_val_clin)
    X_test_clin = scaler.transform(X_test_clin)
    
    print(f"\n✓ Dataset splits:")
    print(f"  Train: {len(X_train_paths)} samples (70%)")
    print(f"  Val:   {len(X_val_paths)} samples (15%)")
    print(f"  Test:  {len(X_test_paths)} samples (15%)")
    
    # Compute class weights to handle imbalanced data
    print("\nStep 3: Computing class weights for imbalanced data...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"✓ Class weights: {class_weight_dict}")
    
    # Create improved generators
    print("\nStep 4: Creating enhanced data generators...")
    train_gen = create_improved_data_generator(
        X_train_paths, X_train_clin, y_train,
        batch_size=batch_size, img_size=(img_size, img_size), 
        augment=True, shuffle=True
    )
    val_gen = create_improved_data_generator(
        X_val_paths, X_val_clin, y_val,
        batch_size=batch_size, img_size=(img_size, img_size), 
        augment=False, shuffle=False
    )
    test_gen = create_improved_data_generator(
        X_test_paths, X_test_clin, y_test,
        batch_size=batch_size, img_size=(img_size, img_size), 
        augment=False, shuffle=False
    )
    
    print(f"✓ Enhanced generators created with strong augmentation")
    print(f"  Training batches: {len(train_gen)}")
    print(f"  Validation batches: {len(val_gen)}")
    print(f"  Test batches: {len(test_gen)}")
    
    # Build model with higher dropout
    print("\nStep 5: Building model with enhanced regularization...")
    detector = MultimodalCancerDetector(
        img_size=(img_size, img_size, 3),
        num_clinical_features=5,
        num_classes=4,
        dropout_rate=dropout_rate  # Increased dropout
    )
    
    model = detector.build_fusion_model()
    
    # Add L2 regularization to dense layers
    print("  Adding L2 regularization to prevent overfitting...")
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = keras.regularizers.l2(0.001)
    
    detector.compile_model(learning_rate=learning_rate)
    
    print("✓ Model built with enhanced regularization")
    detector.get_model_summary()
    
    # Setup improved callbacks
    print("\nStep 6: Setting up callbacks...")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/checkpoints/improved_best_model.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            patience=20,  # Increased patience
            monitor='val_loss',
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # Only stop if no improvement > 0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=7,  # Increased patience
            min_lr=1e-7,
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='logs/improved_training',
            histogram_freq=1
        ),
        # Add learning rate scheduler for gradual decay
        keras.callbacks.LearningRateScheduler(
            lambda epoch: learning_rate * (0.95 ** epoch),
            verbose=0
        )
    ]
    
    print("✓ Callbacks configured with improved settings")
    
    # Train model
    print("\nStep 7: Training model with anti-overfitting techniques...")
    print("="*80)
    
    # Convert generators to tf.data.Dataset for better compatibility
    print("  Converting generators to tf.data.Dataset format...")
    
    def generator_to_dataset(gen):
        """Convert Sequence generator to tf.data.Dataset"""
        def data_generator():
            for i in range(len(gen)):
                (images, clinical), labels = gen[i]
                # Yield each sample individually
                for j in range(len(images)):
                    yield (images[j], clinical[j]), labels[j]
        
        output_signature = (
            (
                tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(5,), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(4,), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=output_signature
        )
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_dataset = generator_to_dataset(train_gen)
    val_dataset = generator_to_dataset(val_gen)
    
    print("  ✓ Datasets created")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Use class weights
        verbose=1
    )
    
    print("\n" + "="*80)
    print("✓ Training completed!")
    print("="*80)
    
    # Evaluate on test set
    print("\nStep 8: Evaluating on test set...")
    test_dataset = generator_to_dataset(test_gen)
    test_results = model.evaluate(test_dataset, verbose=1)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - IMPROVED MODEL")
    print("="*80)
    print(f"Training Samples: {len(X_train_paths)}")
    print(f"Validation Samples: {len(X_val_paths)}")
    print(f"Test Samples: {len(X_test_paths)}")
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    if len(test_results) > 2:
        print(f"Test Precision: {test_results[2]:.4f}")
        print(f"Test Recall: {test_results[3]:.4f}")
        print(f"Test AUC: {test_results[4]:.4f}")
    print("="*80)
    
    # Plot training history
    print("\nStep 9: Generating training plots...")
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(history.history['loss'], label='Train Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Overfitting gap plot
        train_acc = np.array(history.history['accuracy'])
        val_acc = np.array(history.history['val_accuracy'])
        gap = train_acc - val_acc
        axes[1, 1].plot(gap, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--')
        axes[1, 1].set_title('Overfitting Gap (Train - Val Accuracy)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Gap')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('models/checkpoints/improved_training_history.png', dpi=300)
        print("✓ Training plots saved to: models/checkpoints/improved_training_history.png")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
    
    # Save final model
    print("\nStep 10: Saving final model...")
    model.save('models/saved_models/oncovision_improved.keras')
    print("✓ Model saved to: models/saved_models/oncovision_improved.keras")
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'models/saved_models/scaler_improved.pkl')
    print("✓ Scaler saved to: models/saved_models/scaler_improved.pkl")
    
    print("\n✅ SUCCESS! Improved model trained with anti-overfitting techniques!")
    print("\nKey Improvements Applied:")
    print("  ✓ Higher dropout rate (0.5)")
    print("  ✓ Strong data augmentation (rotation, flip, zoom, brightness, noise)")
    print("  ✓ L2 regularization on dense layers")
    print("  ✓ Class weights for imbalanced data")
    print("  ✓ Lower learning rate with gradual decay")
    print("  ✓ Smaller image size for better generalization")
    print("  ✓ Increased early stopping patience")
    
    print("\nNext steps:")
    print("  1. Check training plots: models/checkpoints/improved_training_history.png")
    print("  2. View TensorBoard: tensorboard --logdir logs/improved_training")
    print("  3. Run demo: streamlit run app/demo_app.py")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train improved model with anti-overfitting")
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--img-size', type=int, default=128, help='Image size')
    parser.add_argument('--samples', type=int, default=None, help='Max samples to use (for testing)')
    
    args = parser.parse_args()
    
    print("\n🚀 Starting improved training with anti-overfitting techniques...")
    print(f"   This addresses the overfitting issue you experienced.")
    print(f"   Expected: Better validation accuracy, smaller train-val gap\n")
    
    train_improved_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        img_size=args.img_size,
        max_samples=args.samples
    )
