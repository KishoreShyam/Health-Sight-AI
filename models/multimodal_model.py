"""
Multimodal Fusion Model for OncoVisionAI
Combines CNN (MobileNetV3) for images + MLP for clinical data
"""

import tensorflow as tf
import keras
from keras import layers, Model
from keras.applications import MobileNetV3Small
from typing import Tuple, Optional
import numpy as np


class MultimodalCancerDetector:
    """
    Multimodal AI architecture combining:
    - Image Branch: MobileNetV3-Small (CNN) for lesion images
    - Clinical Branch: MLP for tabular clinical data
    - Fusion Layer: Concatenation + Dense classification
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (224, 224, 3),
        num_clinical_features: int = 5,
        num_classes: int = 4,  # Changed to 4: Normal, BCC, SCC, Melanoma
        dropout_rate: float = 0.3
    ):
        self.img_size = img_size
        self.num_clinical_features = num_clinical_features
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        
        # Class names for multi-class classification
        self.class_names = ['Normal', 'BCC', 'SCC', 'Melanoma']
        self.class_urgency = {
            'Normal': 0,
            'BCC': 1,      # Low urgency - slow growing
            'SCC': 2,      # Medium urgency - can spread
            'Melanoma': 3  # HIGH urgency - aggressive, immediate referral
        }
        
    def build_image_branch(self, trainable_layers: int = 10) -> Model:
        """
        Build CNN branch using MobileNetV3-Small with transfer learning
        
        Args:
            trainable_layers: Number of top layers to fine-tune
        
        Returns:
            Image feature extraction model
        """
        # Load pre-trained MobileNetV3-Small
        base_model = MobileNetV3Small(
            input_shape=self.img_size,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Freeze base layers, unfreeze top layers for fine-tuning
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
        
        # Create image input
        image_input = keras.Input(shape=self.img_size, name='image_input')
        
        # Extract features
        x = base_model(image_input, training=False)
        
        # Additional dense layer for image embeddings
        x = layers.Dense(128, activation='relu', name='image_embedding')(x)
        x = layers.BatchNormalization(name='image_bn')(x)
        x = layers.Dropout(self.dropout_rate, name='image_dropout')(x)
        
        image_branch = Model(inputs=image_input, outputs=x, name='image_branch')
        
        return image_branch
    
    def build_clinical_branch(self) -> Model:
        """
        Build MLP branch for clinical tabular data
        
        Returns:
            Clinical feature extraction model
        """
        # Clinical data input
        clinical_input = keras.Input(
            shape=(self.num_clinical_features,),
            name='clinical_input'
        )
        
        # MLP layers
        x = layers.Dense(64, activation='relu', name='clinical_dense1')(clinical_input)
        x = layers.BatchNormalization(name='clinical_bn1')(x)
        x = layers.Dropout(self.dropout_rate, name='clinical_dropout1')(x)
        
        x = layers.Dense(32, activation='relu', name='clinical_dense2')(x)
        x = layers.BatchNormalization(name='clinical_bn2')(x)
        x = layers.Dropout(self.dropout_rate / 2, name='clinical_dropout2')(x)
        
        # Clinical embeddings
        x = layers.Dense(16, activation='relu', name='clinical_embedding')(x)
        
        clinical_branch = Model(inputs=clinical_input, outputs=x, name='clinical_branch')
        
        return clinical_branch
    
    def build_fusion_model(self) -> Model:
        """
        Build complete multimodal fusion model
        
        Returns:
            Complete multimodal model
        """
        # Build branches
        image_branch = self.build_image_branch()
        clinical_branch = self.build_clinical_branch()
        
        # Inputs
        image_input = keras.Input(shape=self.img_size, name='image_input')
        clinical_input = keras.Input(shape=(self.num_clinical_features,), name='clinical_input')
        
        # Extract features from both branches
        image_features = image_branch(image_input)
        clinical_features = clinical_branch(clinical_input)
        
        # Fusion layer - concatenate embeddings
        fused = layers.Concatenate(name='fusion_layer')([image_features, clinical_features])
        
        # Classification head
        x = layers.Dense(64, activation='relu', name='fusion_dense1')(fused)
        x = layers.BatchNormalization(name='fusion_bn')(x)
        x = layers.Dropout(self.dropout_rate, name='fusion_dropout')(x)
        
        x = layers.Dense(32, activation='relu', name='fusion_dense2')(x)
        
        # Output layer
        output = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        model = Model(
            inputs=[image_input, clinical_input],
            outputs=output,
            name='HealthSightAI_Multimodal'
        )
        
        self.model = model
        return model
    
    def compile_model(
        self,
        learning_rate: float = 1e-4,
        metrics: Optional[list] = None
    ):
        """
        Compile the model with optimizer and loss
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            metrics: List of metrics to track
        """
        if self.model is None:
            self.build_fusion_model()
        
        if metrics is None:
            metrics = [
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=metrics
        )
        
        print("✓ Model compiled successfully")
        return self.model
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_fusion_model()
        
        print("\n" + "="*80)
        print("HEALTH SIGHT AI - MULTIMODAL ARCHITECTURE")
        print("="*80)
        self.model.summary()
        print("="*80)
        
        # Count parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print("="*80 + "\n")
    
    def save_model(self, filepath: str):
        """Save the complete model"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_fusion_model() first.")
        
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")
        return self.model


def create_callbacks(
    model_checkpoint_path: str = 'models/checkpoints/best_model.h5',
    tensorboard_log_dir: str = 'logs',
    early_stopping_patience: int = 10
) -> list:
    """
    Create training callbacks
    
    Args:
        model_checkpoint_path: Path to save best model
        tensorboard_log_dir: Directory for TensorBoard logs
        early_stopping_patience: Patience for early stopping
    
    Returns:
        List of Keras callbacks
    """
    import os
    os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True
        ),
        
        # CSV Logger
        keras.callbacks.CSVLogger(
            filename=os.path.join(tensorboard_log_dir, 'training_log.csv'),
            append=True
        )
    ]
    
    return callbacks


class F1Score(keras.metrics.Metric):
    """Custom F1-Score metric"""
    
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


if __name__ == "__main__":
    # Demo: Build and display model architecture
    print("Building OncoVisionAI Multimodal Model...\n")
    
    detector = MultimodalCancerDetector(
        img_size=(224, 224, 3),
        num_clinical_features=5,
        num_classes=2,
        dropout_rate=0.3
    )
    
    # Build and compile
    model = detector.build_fusion_model()
    detector.compile_model(learning_rate=1e-4)
    
    # Show summary
    detector.get_model_summary()
    
    # Test with dummy data
    print("Testing model with dummy data...")
    dummy_images = np.random.randn(4, 224, 224, 3).astype(np.float32)
    dummy_clinical = np.random.randn(4, 5).astype(np.float32)
    
    predictions = model.predict([dummy_images, dummy_clinical])
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Sample predictions:\n{predictions}")
    
    print("\n✓ Model architecture validated successfully!")
