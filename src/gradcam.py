"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Provides explainability for OncoVisionAI predictions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os


class GradCAM:
    """
    Grad-CAM implementation for visual explanation of CNN predictions
    Shows which regions of the image influenced the model's decision
    """
    
    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to visualize
                       If None, uses the last conv layer
        """
        self.model = model
        self.layer_name = layer_name
        
        if layer_name is None:
            # Find the last convolutional layer
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:  # Conv layer has 4D output
                    self.layer_name = layer.name
                    break
        
        print(f"✓ Grad-CAM initialized with layer: {self.layer_name}")
    
    def compute_heatmap(
        self,
        image: np.ndarray,
        clinical_data: np.ndarray,
        class_idx: Optional[int] = None,
        eps: float = 1e-8
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Grad-CAM heatmap for a given image
        
        Args:
            image: Input image (preprocessed)
            clinical_data: Clinical features
            class_idx: Target class index (if None, uses predicted class)
            eps: Small epsilon for numerical stability
        
        Returns:
            Tuple of (heatmap, prediction_score)
        """
        # Expand dimensions if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        if len(clinical_data.shape) == 1:
            clinical_data = np.expand_dims(clinical_data, axis=0)
        
        # Create a model that outputs both the predictions and the feature maps
        grad_model = keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get the feature maps and predictions
            conv_outputs, predictions = grad_model([image, clinical_data])
            
            # If class_idx is None, use the predicted class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the score for the target class
            class_channel = predictions[:, class_idx]
        
        # Compute gradients of the class score with respect to feature maps
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps by the gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Multiply each feature map by its gradient weight
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average over all feature maps to get the heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Apply ReLU to focus on positive influences
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap to [0, 1]
        heatmap = heatmap / (np.max(heatmap) + eps)
        
        # Get prediction score
        pred_score = float(predictions[0][class_idx])
        
        return heatmap, pred_score
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap
            original_image: Original image (RGB)
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap to use
        
        Returns:
            Superimposed image
        """
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is in correct format
        if original_image.max() <= 1.0:
            original_image = np.uint8(255 * original_image)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed
    
    def generate_explanation(
        self,
        image: np.ndarray,
        clinical_data: np.ndarray,
        original_image: np.ndarray,
        class_names: list = ['Benign', 'Malignant'],
        save_path: Optional[str] = None
    ) -> dict:
        """
        Generate complete visual explanation
        
        Args:
            image: Preprocessed image for model
            clinical_data: Clinical features
            original_image: Original image for visualization
            class_names: List of class names
            save_path: Path to save visualization
        
        Returns:
            Dictionary with heatmap, overlay, and predictions
        """
        # Compute heatmap
        heatmap, pred_score = self.compute_heatmap(image, clinical_data)
        
        # Get full predictions
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        if len(clinical_data.shape) == 1:
            clinical_data = np.expand_dims(clinical_data, axis=0)
        
        predictions = self.model.predict([image, clinical_data], verbose=0)[0]
        pred_class = np.argmax(predictions)
        
        # Create overlay
        overlay = self.overlay_heatmap(heatmap, original_image)
        
        # Create visualization
        if save_path:
            self._create_visualization(
                original_image,
                heatmap,
                overlay,
                predictions,
                class_names,
                clinical_data[0],
                save_path
            )
        
        return {
            'heatmap': heatmap,
            'overlay': overlay,
            'predictions': predictions,
            'predicted_class': pred_class,
            'predicted_label': class_names[pred_class],
            'confidence': float(predictions[pred_class])
        }
    
    def _create_visualization(
        self,
        original: np.ndarray,
        heatmap: np.ndarray,
        overlay: np.ndarray,
        predictions: np.ndarray,
        class_names: list,
        clinical_features: np.ndarray,
        save_path: str
    ):
        """Create and save comprehensive visualization"""
        fig = plt.figure(figsize=(16, 6))
        
        # Original image
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(original)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Heatmap
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(heatmap, cmap='jet')
        ax2.set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Overlay
        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(overlay)
        ax3.set_title('Explainable Prediction', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Predictions and clinical data
        ax4 = plt.subplot(1, 4, 4)
        ax4.axis('off')
        
        # Prediction bars
        y_pos = np.arange(len(class_names))
        colors = ['green' if predictions[i] < 0.5 else 'red' for i in range(len(class_names))]
        
        bars = ax4.barh(y_pos, predictions, color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(class_names)
        ax4.set_xlabel('Confidence', fontweight='bold')
        ax4.set_title('Prediction Results', fontsize=12, fontweight='bold')
        ax4.set_xlim([0, 1])
        
        # Add percentage labels
        for i, (bar, pred) in enumerate(zip(bars, predictions)):
            ax4.text(pred + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{pred*100:.1f}%', va='center', fontweight='bold')
        
        # Add clinical data text
        clinical_text = "\n\nClinical Features:\n" + "-"*25 + "\n"
        feature_names = ['Age', 'Duration (mo)', 'Family Hx', 'Pain', 'Size (mm)']
        for name, value in zip(feature_names, clinical_features):
            clinical_text += f"{name}: {value:.1f}\n"
        
        ax4.text(0.02, -0.5, clinical_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to {save_path}")


def batch_generate_gradcam(
    model: keras.Model,
    images: np.ndarray,
    clinical_data: np.ndarray,
    original_images: np.ndarray,
    output_dir: str = 'outputs/gradcam',
    num_samples: int = 10,
    layer_name: Optional[str] = None
):
    """
    Generate Grad-CAM visualizations for multiple samples
    
    Args:
        model: Trained model
        images: Preprocessed images
        clinical_data: Clinical features
        original_images: Original images for visualization
        output_dir: Directory to save outputs
        num_samples: Number of samples to process
        layer_name: Target layer name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    gradcam = GradCAM(model, layer_name)
    
    num_samples = min(num_samples, len(images))
    
    print(f"\nGenerating Grad-CAM for {num_samples} samples...")
    
    for i in range(num_samples):
        save_path = os.path.join(output_dir, f'gradcam_sample_{i:03d}.png')
        
        result = gradcam.generate_explanation(
            images[i],
            clinical_data[i],
            original_images[i],
            save_path=save_path
        )
        
        print(f"  [{i+1}/{num_samples}] {result['predicted_label']}: "
              f"{result['confidence']*100:.1f}% confidence")
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    print("Grad-CAM Explainability Module for OncoVisionAI")
    print("="*60)
    print("\nThis module provides visual explanations for model predictions")
    print("using Gradient-weighted Class Activation Mapping (Grad-CAM).")
    print("\nUsage:")
    print("  from src.gradcam import GradCAM")
    print("  gradcam = GradCAM(model)")
    print("  result = gradcam.generate_explanation(image, clinical_data, original_image)")
    print("="*60)
