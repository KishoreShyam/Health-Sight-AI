"""
Quick Training Script for Hackathon
Trains on a subset of ISIC data for faster results
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.multimodal_model import MultimodalCancerDetector
from src.data_preprocessing import CancerDataPreprocessor, MultimodalDataGenerator

def train_on_subset(num_samples=5000, epochs=50, batch_size=16):
    """
    Train on a subset of ISIC data for faster training
    
    Args:
        num_samples: Number of samples to use (default 5000)
        epochs: Number of epochs (default 50)
        batch_size: Batch size (default 16)
    """
    
    print("\n" + "="*80)
    print("HEALTH SIGHT AI - SUBSET TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Samples: {num_samples}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Expected Time: 2-4 hours")
    print(f"  Expected Accuracy: 92-95%")
    print("="*80)
    
    # Load metadata
    print("\nStep 1: Loading data...")
    metadata_path = 'data/isic_prepared/prepared_metadata.csv'
    df = pd.read_csv(metadata_path)
    
    print(f"✓ Total available samples: {len(df)}")
    
    # Sample subset (stratified by class)
    print(f"\nStep 2: Sampling {num_samples} samples (stratified)...")
    df_subset = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), num_samples // 4), random_state=42)
    ).reset_index(drop=True)
    
    print(f"✓ Selected {len(df_subset)} samples")
    print("\nClass distribution:")
    for label in sorted(df_subset['label'].unique()):
        count = (df_subset['label'] == label).sum()
        class_name = df_subset[df_subset['label'] == label]['class_name'].iloc[0]
        print(f"  {class_name}: {count}")
    
    # Save subset metadata
    subset_csv = 'data/isic_prepared/subset_metadata.csv'
    df_subset.to_csv(subset_csv, index=False)
    print(f"\n✓ Saved subset to: {subset_csv}")
    
    # Prepare dataset
    print("\nStep 3: Preparing dataset...")
    
    # Load data
    image_paths = []
    labels = []
    clinical_features = []
    
    for idx, row in df_subset.iterrows():
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
    
    # Split data
    from sklearn.preprocessing import StandardScaler
    
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
    
    print(f"✓ Train: {len(X_train_paths)} samples")
    print(f"✓ Val: {len(X_val_paths)} samples")
    print(f"✓ Test: {len(X_test_paths)} samples")
    
    # Create generators
    print("\nStep 4: Creating data generators...")
    train_gen = MultimodalDataGenerator(
        X_train_paths, X_train_clin, y_train,
        batch_size=batch_size, img_size=(128, 128), augment=True, shuffle=True
    )
    val_gen = MultimodalDataGenerator(
        X_val_paths, X_val_clin, y_val,
        batch_size=batch_size, img_size=(128, 128), augment=False, shuffle=False
    )
    test_gen = MultimodalDataGenerator(
        X_test_paths, X_test_clin, y_test,
        batch_size=batch_size, img_size=(128, 128), augment=False, shuffle=False
    )
    
    print(f"✓ Generators created")
    
    # Build model
    print("\nStep 5: Building model...")
    detector = MultimodalCancerDetector(
        img_size=(128, 128, 3),
        num_clinical_features=5,
        num_classes=4,
        dropout_rate=0.4
    )
    
    model = detector.build_fusion_model()
    detector.compile_model(learning_rate=0.0001)
    
    print("✓ Model built and compiled")
    
    # Train
    print("\nStep 6: Training model...")
    print("="*80)
    print("Loading data into memory (5000 samples is manageable)...")
    
    # Load all training data into memory
    print("Loading training data...")
    X_train_img = []
    X_train_clin_list = []
    y_train_list = []
    
    for i in range(len(train_gen)):
        batch = train_gen[i]
        X_train_img.append(batch[0][0])
        X_train_clin_list.append(batch[0][1])
        y_train_list.append(batch[1])
    
    X_train_img = np.concatenate(X_train_img, axis=0)
    X_train_clin_arr = np.concatenate(X_train_clin_list, axis=0)
    y_train_arr = np.concatenate(y_train_list, axis=0)
    
    print(f"✓ Training data loaded: {X_train_img.shape}")
    
    # Load validation data
    print("Loading validation data...")
    X_val_img = []
    X_val_clin_list = []
    y_val_list = []
    
    for i in range(len(val_gen)):
        batch = val_gen[i]
        X_val_img.append(batch[0][0])
        X_val_clin_list.append(batch[0][1])
        y_val_list.append(batch[1])
    
    X_val_img = np.concatenate(X_val_img, axis=0)
    X_val_clin_arr = np.concatenate(X_val_clin_list, axis=0)
    y_val_arr = np.concatenate(y_val_list, axis=0)
    
    print(f"✓ Validation data loaded: {X_val_img.shape}")
    
    import keras
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/checkpoints/subset_best_model.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            patience=10,
            monitor='val_loss',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\nStarting training...")
    # Train with arrays (most reliable for Keras 3.x)
    history = model.fit(
        [X_train_img, X_train_clin_arr],
        y_train_arr,
        validation_data=([X_val_img, X_val_clin_arr], y_val_arr),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*80)
    print("✓ Training completed!")
    print("="*80)
    
    # Evaluate
    print("\nStep 7: Evaluating model...")
    print("Loading test data...")
    X_test_img = []
    X_test_clin_list = []
    y_test_list = []
    
    for i in range(len(test_gen)):
        batch = test_gen[i]
        X_test_img.append(batch[0][0])
        X_test_clin_list.append(batch[0][1])
        y_test_list.append(batch[1])
    
    X_test_img = np.concatenate(X_test_img, axis=0)
    X_test_clin_arr = np.concatenate(X_test_clin_list, axis=0)
    y_test_arr = np.concatenate(y_test_list, axis=0)
    
    print(f"✓ Test data loaded: {X_test_img.shape}")
    
    test_results = model.evaluate([X_test_img, X_test_clin_arr], y_test_arr, verbose=1)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
    print("="*80)
    
    # Save final model
    print("\nStep 8: Saving model...")
    model.save('models/saved_models/oncovision_multimodal.keras')
    print("✓ Model saved to: models/saved_models/oncovision_multimodal.keras")
    
    print("\n✅ SUCCESS! Your model is ready for the demo!")
    print("\nNext steps:")
    print("  1. Run demo: streamlit run app/demo_app.py")
    print("  2. Test predictions on new images")
    print("  3. Present your results!")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train on ISIC subset")
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    train_on_subset(
        num_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
