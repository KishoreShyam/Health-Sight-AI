"""
Complete Pipeline Runner for OncoVisionAI
Executes the full workflow from data preparation to model deployment
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"▶ {description}...")
    print(f"  Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully!\n")


def main(args):
    """Run the complete pipeline"""
    start_time = datetime.now()
    
    print_header("HEALTH SIGHT AI - COMPLETE PIPELINE")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Quantization: {args.quantize}")
    
    # Step 1: Data Preparation
    if not args.skip_data:
        print_header("STEP 1: DATA PREPARATION")
        cmd = f"python src/data_preprocessing.py --prepare-all --num-samples {args.num_samples}"
        run_command(cmd, "Data preparation")
    else:
        print_header("STEP 1: DATA PREPARATION (SKIPPED)")
    
    # Step 2: Model Training
    if not args.skip_training:
        print_header("STEP 2: MODEL TRAINING")
        cmd = (f"python models/train.py "
               f"--epochs {args.epochs} "
               f"--batch-size {args.batch_size} "
               f"--learning-rate {args.learning_rate} "
               f"--img-size {args.img_size}")
        run_command(cmd, "Model training")
    else:
        print_header("STEP 2: MODEL TRAINING (SKIPPED)")
    
    # Step 3: TFLite Export
    if not args.skip_export:
        print_header("STEP 3: TFLITE EXPORT & OPTIMIZATION")
        benchmark_flag = "--benchmark" if args.benchmark else ""
        cmd = (f"python models/export_tflite.py "
               f"--quantize {args.quantize} "
               f"{benchmark_flag}")
        run_command(cmd, "TFLite export and quantization")
    else:
        print_header("STEP 3: TFLITE EXPORT (SKIPPED)")
    
    # Step 4: Generate Visualizations
    if not args.skip_viz:
        print_header("STEP 4: GENERATING VISUALIZATIONS")
        print("▶ Creating demo visualizations...")
        
        # Create demo script
        viz_script = """
import sys
sys.path.append('.')
from src.utils import create_demo_images, print_system_info
print_system_info()
create_demo_images(num_images=20)
print('\\n✓ Visualizations created!')
"""
        with open('temp_viz.py', 'w') as f:
            f.write(viz_script)
        
        run_command("python temp_viz.py", "Visualization generation")
        
        # Cleanup
        if os.path.exists('temp_viz.py'):
            os.remove('temp_viz.py')
    else:
        print_header("STEP 4: VISUALIZATIONS (SKIPPED)")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Start time:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    
    print("\n📁 Generated Files:")
    print("  ├── data/clinical_data.csv")
    print("  ├── data/raw/images/")
    print("  ├── models/saved_models/oncovision_multimodal.keras")
    print("  ├── models/tflite/oncovision_quantized.tflite")
    print("  ├── outputs/training_history.png")
    print("  └── outputs/evaluation_results.txt")
    
    print("\n🚀 Next Steps:")
    print("  1. Review training results:")
    print("     - Check outputs/training_history.png")
    print("     - Read outputs/evaluation_results.txt")
    print("\n  2. Test the model:")
    print("     - Run: streamlit run app/demo_app.py")
    print("\n  3. Deploy to mobile:")
    print("     - Use models/tflite/oncovision_quantized.tflite")
    print("     - Integrate into Flutter/React Native app")
    
    print("\n" + "="*80)
    print("  OncoVisionAI is ready for deployment! 🏥")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete OncoVisionAI pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults
  python run_pipeline.py

  # Quick test with minimal data
  python run_pipeline.py --num-samples 100 --epochs 5

  # Production training
  python run_pipeline.py --num-samples 5000 --epochs 50 --batch-size 64

  # Skip data preparation (use existing data)
  python run_pipeline.py --skip-data

  # Only train model
  python run_pipeline.py --skip-data --skip-export --skip-viz
        """
    )
    
    # Data arguments
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of sample images to generate (default: 1000)')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data preparation step')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (default: 0.0001)')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size (default: 224)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step')
    
    # Export arguments
    parser.add_argument('--quantize', type=str, default='full',
                       choices=['none', 'dynamic', 'float16', 'int8', 'full'],
                       help='Quantization type (default: full)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmark after export')
    parser.add_argument('--skip-export', action='store_true',
                       help='Skip TFLite export step')
    
    # Visualization arguments
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed with error: {str(e)}")
        sys.exit(1)
