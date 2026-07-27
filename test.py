import sys
import os
from azure_vision import analyze_image

def main():
    image_path = "test.jpg"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        print("Please place a test image named 'test.jpg' in the root directory or pass the image path as an argument.")
        sys.exit(1)
        
    print(f"Analyzing {image_path} using Azure Vision API...")
    try:
        result = analyze_image(image_path)
        
        print("\n--- Caption ---")
        if result.caption:
            print(f"Text: {result.caption.text}")
            print(f"Confidence: {result.caption.confidence:.4f}")
        else:
            print("No caption generated (unsupported in this region or API returned none).")

        print("\n--- Tags ---")
        if result.tags and result.tags.list:
            for tag in result.tags.list:
                print(f"- {tag.name} (confidence: {tag.confidence:.4f})")
        else:
            print("No tags detected.")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your .env file is correctly configured with valid AZURE_ENDPOINT and AZURE_KEY.")

if __name__ == "__main__":
    main()
