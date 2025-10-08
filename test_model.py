"""
Quick Model Testing Script for OncoVisionAI
Tests the trained model with sample predictions
"""

import os
import sys
import numpy as np
import cv2
from tensorflow import keras
import argparse


def test_model_inference(model_path: str, num_tests: int = 5):
    """
    Test model with random inputs
    
    Args:
        model_path: Path to trained model
        num_tests: Number of test inferences to run
    """
    print("\n" + "="*80)
    print("ONCO VISION AI - MODEL INFERENCE TEST")
    print("="*80)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    try:
        model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return
    
    # Model info
    print(f"\nModel Information:")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Input shapes: {[inp.shape for inp in model.inputs]}")
    print(f"  Output shape: {model.output.shape}")
    
    # Test predictions
    print(f"\nRunning {num_tests} test inferences...")
    print("-" * 80)
    
    for i in range(num_tests):
        # Create dummy inputs
        dummy_image = np.random.randn(1, 224, 224, 3).astype(np.float32)
        dummy_clinical = np.random.randn(1, 5).astype(np.float32)
        
        # Predict
        predictions = model.predict([dummy_image, dummy_clinical], verbose=0)
        
        # Display results
        benign_prob = predictions[0][0] * 100
        malignant_prob = predictions[0][1] * 100
        predicted_class = "Malignant" if malignant_prob > benign_prob else "Benign"
        
        print(f"Test {i+1}:")
        print(f"  Benign:    {benign_prob:6.2f}%")
        print(f"  Malignant: {malignant_prob:6.2f}%")
        print(f"  → Predicted: {predicted_class}")
        print()
    
    print("-" * 80)
    print("✓ All inference tests completed successfully!")
    print("="*80)


def test_with_real_image(model_path: str, image_path: str):
    """
    Test model with a real image
    
    Args:
        model_path: Path to trained model
        image_path: Path to test image
    """
    print("\n" + "="*80)
    print("TESTING WITH REAL IMAGE")
    print("="*80)
    
    # Load model
    print(f"\nLoading model...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded")
    
    # Load and preprocess image
    print(f"\nLoading image from {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"✗ Error: Image not found at {image_path}")
        return
    
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {img.shape}")
    
    # Preprocess
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Create sample clinical data
    clinical_data = np.array([[
        55.0,   # Age
        8.0,    # Symptom duration (months)
        1.0,    # Family history (yes)
        6.0,    # Pain score
        15.0    # Lesion size (mm)
    ]])
    
    # Normalize clinical data (approximate)
    clinical_mean = np.array([50.0, 10.0, 0.5, 5.0, 15.0])
    clinical_std = np.array([18.0, 10.0, 0.5, 3.0, 10.0])
    clinical_normalized = (clinical_data - clinical_mean) / clinical_std
    
    print("\nClinical Data:")
    print(f"  Age: {clinical_data[0][0]:.0f} years")
    print(f"  Duration: {clinical_data[0][1]:.0f} months")
    print(f"  Family History: {'Yes' if clinical_data[0][2] > 0.5 else 'No'}")
    print(f"  Pain Score: {clinical_data[0][3]:.0f}/10")
    print(f"  Lesion Size: {clinical_data[0][4]:.0f} mm")
    
    # Predict
    print("\nRunning inference...")
    predictions = model.predict([img_batch, clinical_normalized], verbose=0)
    
    # Display results
    benign_prob = predictions[0][0] * 100
    malignant_prob = predictions[0][1] * 100
    predicted_class = "Malignant" if malignant_prob > benign_prob else "Benign"
    confidence = max(benign_prob, malignant_prob)
    
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    print(f"\n  Benign Probability:    {benign_prob:6.2f}%")
    print(f"  Malignant Probability: {malignant_prob:6.2f}%")
    print(f"\n  → PREDICTION: {predicted_class}")
    print(f"  → CONFIDENCE: {confidence:.2f}%")
    print("\n" + "="*80)
    
    # Recommendation
    if predicted_class == "Malignant":
        print("\n⚠️  RECOMMENDATION: Immediate consultation with oncologist recommended")
    else:
        print("\n✓ RECOMMENDATION: Appears benign, but consult healthcare professional")
    
    print("="*80)


def benchmark_inference_speed(model_path: str, num_runs: int = 100):
    """
    Benchmark model inference speed
    
    Args:
        model_path: Path to trained model
        num_runs: Number of inference runs
    """
    import time
    
    print("\n" + "="*80)
    print("INFERENCE SPEED BENCHMARK")
    print("="*80)
    
    # Load model
    print(f"\nLoading model...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded")
    
    # Create dummy inputs
    dummy_image = np.random.randn(1, 224, 224, 3).astype(np.float32)
    dummy_clinical = np.random.randn(1, 5).astype(np.float32)
    
    # Warmup
    print(f"\nWarming up (10 runs)...")
    for _ in range(10):
        _ = model.predict([dummy_image, dummy_clinical], verbose=0)
    
    # Benchmark
    print(f"Running benchmark ({num_runs} runs)...")
    times = []
    
    for i in range(num_runs):
        start = time.time()
        _ = model.predict([dummy_image, dummy_clinical], verbose=0)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_runs} runs")
    
    # Calculate statistics
    times = np.array(times)
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"\nNumber of runs: {num_runs}")
    print(f"Mean inference time: {np.mean(times):.2f} ms")
    print(f"Std deviation: {np.std(times):.2f} ms")
    print(f"Min time: {np.min(times):.2f} ms")
    print(f"Max time: {np.max(times):.2f} ms")
    print(f"Median time: {np.median(times):.2f} ms")
    print(f"95th percentile: {np.percentile(times, 95):.2f} ms")
    print("="*80)
    
    # Performance assessment
    mean_time = np.mean(times)
    if mean_time < 500:
        print("\n✓ EXCELLENT: Model meets mobile deployment target (< 500ms)")
    elif mean_time < 1000:
        print("\n✓ GOOD: Model suitable for mobile deployment")
    else:
        print("\n⚠️  WARNING: Model may be slow on low-end devices")
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OncoVisionAI Model")
    
    parser.add_argument('--model-path', type=str,
                       default='models/saved_models/oncovision_multimodal.h5',
                       help='Path to trained model')
    parser.add_argument('--test-type', type=str, default='inference',
                       choices=['inference', 'image', 'benchmark'],
                       help='Type of test to run')
    parser.add_argument('--image-path', type=str,
                       help='Path to test image (for image test)')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test inferences')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"\n✗ Error: Model not found at {args.model_path}")
        print(f"\nPlease train the model first:")
        print(f"  python models/train.py")
        print(f"\nOr run the complete pipeline:")
        print(f"  python run_pipeline.py")
        sys.exit(1)
    
    # Run appropriate test
    if args.test_type == 'inference':
        test_model_inference(args.model_path, args.num_tests)
    
    elif args.test_type == 'image':
        if not args.image_path:
            print("\n✗ Error: --image-path required for image test")
            sys.exit(1)
        test_with_real_image(args.model_path, args.image_path)
    
    elif args.test_type == 'benchmark':
        benchmark_inference_speed(args.model_path, args.num_runs)
    
    print("\n✓ Testing complete!\n")
