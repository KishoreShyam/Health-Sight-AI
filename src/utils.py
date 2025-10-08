"""
Utility Functions for OncoVisionAI
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json
from typing import Dict, List, Tuple
import cv2


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Benign', 'Malignant'],
    save_path: str = 'outputs/confusion_matrix.png',
    normalize: bool = True
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize values
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = 'outputs/roc_curve.png'
):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels (binary)
        y_scores: Prediction scores for positive class
        save_path: Path to save plot
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curve saved to {save_path}")
    print(f"  AUC Score: {roc_auc:.4f}")
    
    return roc_auc


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Benign', 'Malignant'],
    save_path: str = 'outputs/classification_report.txt'
):
    """
    Generate and save classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save report
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("OncoVisionAI - Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"\n✓ Classification report saved to {save_path}")


def visualize_sample_predictions(
    images: np.ndarray,
    clinical_data: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    num_samples: int = 12,
    save_path: str = 'outputs/sample_predictions.png'
):
    """
    Visualize sample predictions
    
    Args:
        images: Input images
        clinical_data: Clinical features
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    num_samples = min(num_samples, len(images))
    
    # Create grid
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    class_names = ['Benign', 'Malignant']
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        img = images[i]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Denormalize if needed
        if img.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        
        # Create title with prediction info
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        confidence = y_scores[i][y_pred[i]] * 100
        
        correct = y_true[i] == y_pred[i]
        color = 'green' if correct else 'red'
        
        title = f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)"
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Sample predictions saved to {save_path}")


def save_model_metadata(
    model_info: Dict,
    save_path: str = 'models/model_metadata.json'
):
    """
    Save model metadata
    
    Args:
        model_info: Dictionary with model information
        save_path: Path to save metadata
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"✓ Model metadata saved to {save_path}")


def calculate_model_size(model_path: str) -> Dict[str, float]:
    """
    Calculate model file size
    
    Args:
        model_path: Path to model file
    
    Returns:
        Dictionary with size information
    """
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    
    return {
        'bytes': size_bytes,
        'kilobytes': size_kb,
        'megabytes': size_mb
    }


def create_comparison_table(
    baseline_metrics: Dict,
    multimodal_metrics: Dict,
    save_path: str = 'outputs/model_comparison.txt'
):
    """
    Create comparison table between baseline and multimodal models
    
    Args:
        baseline_metrics: Metrics for baseline (image-only) model
        multimodal_metrics: Metrics for multimodal model
        save_path: Path to save comparison
    """
    comparison = []
    comparison.append("="*80)
    comparison.append("MODEL COMPARISON: Image-Only vs Multimodal")
    comparison.append("="*80)
    comparison.append("")
    comparison.append(f"{'Metric':<20} {'Image-Only':<15} {'Multimodal':<15} {'Improvement':<15}")
    comparison.append("-"*80)
    
    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        multimodal_val = multimodal_metrics[metric]
        improvement = ((multimodal_val - baseline_val) / baseline_val) * 100
        
        comparison.append(
            f"{metric.capitalize():<20} "
            f"{baseline_val:<15.4f} "
            f"{multimodal_val:<15.4f} "
            f"{improvement:>+14.2f}%"
        )
    
    comparison.append("="*80)
    
    # Print
    for line in comparison:
        print(line)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(comparison))
    
    print(f"\n✓ Comparison table saved to {save_path}")


def create_demo_images(output_dir: str = 'data/demo_images', num_images: int = 10):
    """
    Create demo images for testing
    
    Args:
        output_dir: Directory to save demo images
        num_images: Number of demo images to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_images} demo images...")
    
    for i in range(num_images):
        # Create synthetic lesion image
        img = np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
        
        # Add lesion-like structure
        center = (200, 200)
        radius = np.random.randint(40, 100)
        
        # Irregular shape
        for _ in range(5):
            offset_x = np.random.randint(-30, 30)
            offset_y = np.random.randint(-30, 30)
            r = np.random.randint(radius-20, radius+20)
            color = tuple(np.random.randint(50, 150, 3).tolist())
            cv2.circle(img, (center[0]+offset_x, center[1]+offset_y), r, color, -1)
        
        # Add texture
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add border irregularity
        if i % 2 == 1:  # Malignant-like
            for _ in range(10):
                pt1 = (center[0] + np.random.randint(-radius, radius),
                       center[1] + np.random.randint(-radius, radius))
                pt2 = (center[0] + np.random.randint(-radius, radius),
                       center[1] + np.random.randint(-radius, radius))
                color = tuple(np.random.randint(30, 100, 3).tolist())
                cv2.line(img, pt1, pt2, color, 2)
        
        # Save
        label = 'malignant' if i % 2 == 1 else 'benign'
        filename = f'demo_{i:02d}_{label}.jpg'
        cv2.imwrite(os.path.join(output_dir, filename), img)
    
    print(f"✓ Demo images created in {output_dir}/")


def print_system_info():
    """Print system and library information"""
    import tensorflow as tf
    import platform
    
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    print("OncoVisionAI Utility Functions")
    print("="*60)
    print("\nAvailable utilities:")
    print("  - plot_confusion_matrix()")
    print("  - plot_roc_curve()")
    print("  - generate_classification_report()")
    print("  - visualize_sample_predictions()")
    print("  - save_model_metadata()")
    print("  - create_comparison_table()")
    print("  - create_demo_images()")
    print("  - print_system_info()")
    print("="*60)
    
    # Print system info
    print_system_info()
