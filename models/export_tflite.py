"""
TensorFlow Lite Export and Quantization for OncoVisionAI
Optimizes model for mobile deployment with extreme compression
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TFLiteConverter:
    """
    Convert and optimize Keras model to TensorFlow Lite
    Supports multiple quantization strategies for extreme optimization
    """
    
    def __init__(self, model_path: str):
        """
        Initialize converter
        
        Args:
            model_path: Path to saved Keras model (.h5)
        """
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
    
    def convert_to_tflite(
        self,
        output_path: str,
        quantization: str = 'none'
    ) -> str:
        """
        Convert model to TFLite format
        
        Args:
            output_path: Path to save TFLite model
            quantization: Quantization type ('none', 'dynamic', 'float16', 'int8')
        
        Returns:
            Path to saved TFLite model
        """
        print(f"\nConverting model to TFLite with {quantization} quantization...")
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply quantization
        if quantization == 'dynamic':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("  Using dynamic range quantization")
        
        elif quantization == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            print("  Using float16 quantization")
        
        elif quantization == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # For full integer quantization, we need a representative dataset
            print("  Using int8 quantization (requires representative dataset)")
            print("  Note: This will be applied if representative dataset is provided")
        
        else:
            print("  No quantization applied")
        
        # Convert
        try:
            tflite_model = converter.convert()
            
            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get file sizes
            original_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
            tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            compression_ratio = original_size / tflite_size
            
            print(f"\n✓ Conversion successful!")
            print(f"  Original model size: {original_size:.2f} MB")
            print(f"  TFLite model size: {tflite_size:.2f} MB")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            print(f"  Saved to: {output_path}")
            
            return output_path
        
        except Exception as e:
            print(f"\n✗ Conversion failed: {str(e)}")
            raise
    
    def convert_with_full_quantization(
        self,
        output_path: str,
        representative_dataset_gen
    ) -> str:
        """
        Convert with full integer quantization (8-bit)
        This provides maximum optimization for mobile devices
        
        Args:
            output_path: Path to save TFLite model
            representative_dataset_gen: Generator function for representative data
        
        Returns:
            Path to saved TFLite model
        """
        print("\nConverting with FULL INTEGER QUANTIZATION (8-bit)...")
        print("This is the most optimized format for low-end devices!")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Enable full integer quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        
        # Ensure all ops are quantized to int8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        try:
            tflite_model = converter.convert()
            
            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get file sizes
            original_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
            tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            compression_ratio = original_size / tflite_size
            
            print(f"\n✓ Full quantization successful!")
            print(f"  Original model size: {original_size:.2f} MB")
            print(f"  Quantized TFLite size: {tflite_size:.2f} MB")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            print(f"  Model is now optimized for low-end smartphones!")
            print(f"  Saved to: {output_path}")
            
            return output_path
        
        except Exception as e:
            print(f"\n✗ Full quantization failed: {str(e)}")
            raise


class TFLiteInference:
    """Run inference with TFLite model"""
    
    def __init__(self, tflite_model_path: str):
        """
        Initialize TFLite interpreter
        
        Args:
            tflite_model_path: Path to TFLite model
        """
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✓ TFLite model loaded from {tflite_model_path}")
        print(f"  Input shapes: {[inp['shape'] for inp in self.input_details]}")
        print(f"  Output shape: {self.output_details[0]['shape']}")
    
    def predict(self, image: np.ndarray, clinical_data: np.ndarray) -> np.ndarray:
        """
        Run inference
        
        Args:
            image: Input image
            clinical_data: Clinical features
        
        Returns:
            Predictions
        """
        # Prepare inputs
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        if len(clinical_data.shape) == 1:
            clinical_data = np.expand_dims(clinical_data, axis=0)
        
        # Convert to appropriate dtype
        image = image.astype(self.input_details[0]['dtype'])
        clinical_data = clinical_data.astype(self.input_details[1]['dtype'])
        
        # Set inputs
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.set_tensor(self.input_details[1]['index'], clinical_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output
    
    def benchmark(self, num_runs: int = 100) -> dict:
        """
        Benchmark inference speed
        
        Args:
            num_runs: Number of inference runs
        
        Returns:
            Benchmark results
        """
        import time
        
        # Create dummy inputs
        image_shape = self.input_details[0]['shape']
        clinical_shape = self.input_details[1]['shape']
        
        dummy_image = np.random.randn(*image_shape).astype(self.input_details[0]['dtype'])
        dummy_clinical = np.random.randn(*clinical_shape).astype(self.input_details[1]['dtype'])
        
        # Warmup
        for _ in range(10):
            self.predict(dummy_image[0], dummy_clinical[0])
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.predict(dummy_image[0], dummy_clinical[0])
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        results = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times)
        }
        
        print("\n" + "="*60)
        print("INFERENCE BENCHMARK RESULTS")
        print("="*60)
        print(f"Number of runs: {num_runs}")
        print(f"Mean inference time: {results['mean_ms']:.2f} ms")
        print(f"Std deviation: {results['std_ms']:.2f} ms")
        print(f"Min time: {results['min_ms']:.2f} ms")
        print(f"Max time: {results['max_ms']:.2f} ms")
        print(f"Median time: {results['median_ms']:.2f} ms")
        print("="*60)
        
        return results


def create_representative_dataset(data_dir: str, num_samples: int = 100):
    """
    Create representative dataset generator for quantization
    
    Args:
        data_dir: Directory containing sample data
        num_samples: Number of samples to use
    
    Returns:
        Generator function
    """
    from src.data_preprocessing import CancerDataPreprocessor
    import pandas as pd
    
    # Load some sample data
    preprocessor = CancerDataPreprocessor(img_size=(224, 224))
    
    # Try to load clinical data
    clinical_csv = os.path.join(data_dir, 'clinical_data.csv')
    if os.path.exists(clinical_csv):
        df = pd.read_csv(clinical_csv)
        df = df.head(num_samples)
        
        def representative_dataset_gen():
            for idx, row in df.iterrows():
                # Load image
                img_path = os.path.join(data_dir, 'raw/images', row['image_id'])
                if os.path.exists(img_path):
                    img = preprocessor.load_and_preprocess_image(img_path, augment=False)
                    img = np.expand_dims(img, axis=0).astype(np.float32)
                    
                    # Get clinical features
                    clinical = np.array([[
                        row['age'],
                        row['symptom_duration_months'],
                        row['family_history'],
                        row['pain_score'],
                        row['lesion_size_mm']
                    ]], dtype=np.float32)
                    
                    yield [img, clinical]
    else:
        # Generate dummy data
        def representative_dataset_gen():
            for _ in range(num_samples):
                img = np.random.randn(1, 224, 224, 3).astype(np.float32)
                clinical = np.random.randn(1, 5).astype(np.float32)
                yield [img, clinical]
    
    return representative_dataset_gen


def main(args):
    """Main conversion pipeline"""
    print("\\n" + "="*80)
    print("HEALTH SIGHT AI - TFLITE EXPORT & OPTIMIZATION")
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Quantization: {args.quantize}")
    print("="*80 + "\n")
    
    # Create converter
    converter = TFLiteConverter(args.model_path)
    
    # Convert based on quantization type
    if args.quantize == 'full':
        # Full integer quantization
        print("Preparing representative dataset...")
        rep_dataset_gen = create_representative_dataset(
            args.data_dir,
            num_samples=args.num_representative_samples
        )
        
        tflite_path = converter.convert_with_full_quantization(
            args.output_path,
            rep_dataset_gen
        )
    else:
        # Standard conversion
        tflite_path = converter.convert_to_tflite(
            args.output_path,
            quantization=args.quantize
        )
    
    # Benchmark if requested
    if args.benchmark:
        print("\nRunning inference benchmark...")
        inferencer = TFLiteInference(tflite_path)
        inferencer.benchmark(num_runs=args.benchmark_runs)
    
    print("\n✓ Export complete!")
    print(f"\nTFLite model ready for mobile deployment: {tflite_path}")
    print("\nNext steps:")
    print("  1. Test the model: python models/export_tflite.py --test")
    print("  2. Integrate into Flutter/React Native app")
    print("  3. Deploy to Android/iOS devices")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OncoVisionAI to TFLite")
    
    parser.add_argument('--model-path', type=str,
                       default='models/saved_models/oncovision_multimodal.keras',
                       help='Path to trained Keras model')
    parser.add_argument('--output-path', type=str,
                       default='models/tflite/oncovision_quantized.tflite',
                       help='Path to save TFLite model')
    parser.add_argument('--quantize', type=str, default='full',
                       choices=['none', 'dynamic', 'float16', 'int8', 'full'],
                       help='Quantization type')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory for representative dataset')
    parser.add_argument('--num-representative-samples', type=int, default=100,
                       help='Number of samples for representative dataset')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmark')
    parser.add_argument('--benchmark-runs', type=int, default=100,
                       help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    main(args)
