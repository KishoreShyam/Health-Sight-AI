"""
Data Preprocessing Module for OncoVisionAI
Handles image preprocessing, augmentation, and clinical data generation
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tensorflow as tf
from typing import Tuple, Dict, List
import json


class CancerDataPreprocessor:
    """Preprocessor for multimodal cancer detection data"""
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224)):
        self.img_size = img_size
        self.scaler = StandardScaler()
        
    def get_augmentation_pipeline(self, mode: str = 'train') -> A.Compose:
        """
        Create augmentation pipeline for training/validation
        
        Args:
            mode: 'train' or 'val'
        
        Returns:
            Albumentations composition
        """
        if mode == 'train':
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.5
                ),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0
                    ),
                ], p=0.7),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.3),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    def load_and_preprocess_image(self, image_path: str, augment: bool = False) -> np.ndarray:
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image file
            augment: Whether to apply augmentation
        
        Returns:
            Preprocessed image array
        """
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        transform = self.get_augmentation_pipeline('train' if augment else 'val')
        augmented = transform(image=img)
        img = augmented['image']
        
        return img
    
    def generate_clinical_data(
        self,
        num_samples: int,
        image_paths: List[str],
        labels: List[int],
        output_path: str = 'data/clinical_data.csv'
    ) -> pd.DataFrame:
        """
        Generate simulated clinical triage data for multimodal learning
        
        Features:
        - Age (years)
        - Symptom Duration (months)
        - Family History (0/1)
        - Pain Score (0-10)
        - Lesion Size (mm)
        
        Args:
            num_samples: Number of samples
            image_paths: List of image file paths
            labels: List of labels (0=benign, 1=malignant)
            output_path: Path to save CSV
        
        Returns:
            DataFrame with clinical features
        """
        np.random.seed(42)
        
        clinical_data = []
        
        for i in range(num_samples):
            label = labels[i] if i < len(labels) else np.random.randint(0, 2)
            
            # Generate correlated features (malignant cases tend to have worse indicators)
            if label == 1:  # Malignant
                age = np.random.normal(60, 15)
                duration = np.random.exponential(12)
                family_history = np.random.choice([0, 1], p=[0.4, 0.6])
                pain_score = np.random.normal(6, 2)
                lesion_size = np.random.normal(15, 5)
            else:  # Benign
                age = np.random.normal(45, 18)
                duration = np.random.exponential(6)
                family_history = np.random.choice([0, 1], p=[0.7, 0.3])
                pain_score = np.random.normal(3, 2)
                lesion_size = np.random.normal(8, 3)
            
            # Clip values to realistic ranges
            age = np.clip(age, 18, 95)
            duration = np.clip(duration, 0.5, 60)
            pain_score = np.clip(pain_score, 0, 10)
            lesion_size = np.clip(lesion_size, 2, 50)
            
            clinical_data.append({
                'image_id': os.path.basename(image_paths[i]) if i < len(image_paths) else f'img_{i:05d}.jpg',
                'age': round(age, 1),
                'symptom_duration_months': round(duration, 1),
                'family_history': int(family_history),
                'pain_score': round(pain_score, 1),
                'lesion_size_mm': round(lesion_size, 1),
                'label': label
            })
        
        df = pd.DataFrame(clinical_data)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Clinical data saved to {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Benign: {(df['label']==0).sum()}, Malignant: {(df['label']==1).sum()}")
        
        return df
    
    def prepare_multimodal_dataset(
        self,
        image_dir: str,
        clinical_csv: str,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict:
        """
        Prepare complete multimodal dataset with train/val/test splits
        
        Args:
            image_dir: Directory containing images
            clinical_csv: Path to clinical data CSV
            test_size: Proportion for test set
            val_size: Proportion for validation set
        
        Returns:
            Dictionary with train/val/test data
        """
        # Load clinical data
        clinical_df = pd.read_csv(clinical_csv)
        
        # Get image paths
        image_paths = []
        labels = []
        clinical_features = []
        
        for idx, row in clinical_df.iterrows():
            img_path = os.path.join(image_dir, row['image_id'])
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
        
        # Convert to arrays
        labels = np.array(labels)
        clinical_features = np.array(clinical_features)
        
        # Split data
        X_train_paths, X_temp_paths, y_train, y_temp, X_train_clinical, X_temp_clinical = \
            train_test_split(image_paths, labels, clinical_features, 
                           test_size=test_size + val_size, random_state=42, stratify=labels)
        
        val_ratio = val_size / (test_size + val_size)
        X_val_paths, X_test_paths, y_val, y_test, X_val_clinical, X_test_clinical = \
            train_test_split(X_temp_paths, y_temp, X_temp_clinical,
                           test_size=(1-val_ratio), random_state=42, stratify=y_temp)
        
        # Normalize clinical features
        X_train_clinical = self.scaler.fit_transform(X_train_clinical)
        X_val_clinical = self.scaler.transform(X_val_clinical)
        X_test_clinical = self.scaler.transform(X_test_clinical)
        
        dataset = {
            'train': {
                'image_paths': X_train_paths,
                'clinical': X_train_clinical,
                'labels': y_train
            },
            'val': {
                'image_paths': X_val_paths,
                'clinical': X_val_clinical,
                'labels': y_val
            },
            'test': {
                'image_paths': X_test_paths,
                'clinical': X_test_clinical,
                'labels': y_test
            }
        }
        
        print("\n✓ Dataset prepared:")
        print(f"  Train: {len(X_train_paths)} samples")
        print(f"  Val:   {len(X_val_paths)} samples")
        print(f"  Test:  {len(X_test_paths)} samples")
        
        return dataset


class MultimodalDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for multimodal inputs"""
    
    def __init__(
        self,
        image_paths: List[str],
        clinical_features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        img_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        shuffle: bool = True
    ):
        super().__init__()
        self.image_paths = image_paths
        self.clinical_features = clinical_features
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.preprocessor = CancerDataPreprocessor(img_size)
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X_images = np.zeros((len(batch_indexes), *self.img_size, 3), dtype=np.float32)
        X_clinical = np.zeros((len(batch_indexes), self.clinical_features.shape[1]), dtype=np.float32)
        y = np.zeros((len(batch_indexes), 4), dtype=np.float32)  # 4 classes: Normal, BCC, SCC, Melanoma
        
        for i, idx in enumerate(batch_indexes):
            # Load and preprocess image
            X_images[i] = self.preprocessor.load_and_preprocess_image(
                self.image_paths[idx],
                augment=self.augment
            )
            
            # Get clinical features
            X_clinical[i] = self.clinical_features[idx]
            
            # One-hot encode label
            y[i, self.labels[idx]] = 1
        
        return [X_images, X_clinical], y
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)


def create_sample_dataset(output_dir: str = 'data', num_samples: int = 1000):
    """
    Create a sample dataset for testing (generates dummy images)
    
    Args:
        output_dir: Directory to save sample data
        num_samples: Number of sample images to generate
    """
    os.makedirs(f"{output_dir}/raw/images", exist_ok=True)
    
    print(f"Generating {num_samples} sample images...")
    
    image_paths = []
    labels = []
    
    for i in range(num_samples):
        # Generate random image (simulating skin lesion)
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Add some structure (circle to simulate lesion)
        center = (150, 150)
        radius = np.random.randint(30, 80)
        color = tuple(np.random.randint(50, 200, 3).tolist())
        cv2.circle(img, center, radius, color, -1)
        
        # Add noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Save image with 4-class labels: 0=Normal, 1=BCC, 2=SCC, 3=Melanoma
        label = np.random.randint(0, 4)
        class_names = ['normal', 'bcc', 'scc', 'melanoma']
        img_name = f"sample_{i:05d}_{class_names[label]}.jpg"
        img_path = f"{output_dir}/raw/images/{img_name}"
        cv2.imwrite(img_path, img)
        
        image_paths.append(img_path)
        labels.append(label)
    
    print(f"✓ Generated {num_samples} sample images")
    
    # Generate clinical data
    preprocessor = CancerDataPreprocessor()
    clinical_df = preprocessor.generate_clinical_data(
        num_samples,
        image_paths,
        labels,
        output_path=f"{output_dir}/clinical_data.csv"
    )
    
    print(f"\n✓ Sample dataset created in '{output_dir}/'")
    return image_paths, labels, clinical_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data preprocessing for OncoVisionAI")
    parser.add_argument('--prepare-all', action='store_true', help='Create sample dataset')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    if args.prepare_all:
        create_sample_dataset(args.output_dir, args.num_samples)
        print("\n✓ Data preparation complete!")
    else:
        print("Use --prepare-all to create sample dataset")
