import sys
import os
from azure_blob import upload_blob

def main():
    test_file = "test_blob.txt"
    with open(test_file, "w") as f:
        f.write("Hello, Health Sight AI Blob Storage test!")
        
    print(f"Uploading {test_file} to Azure Blob Storage...")
    try:
        url = upload_blob(test_file, test_file)
        print(f"\nUpload successful!")
        print(f"Public URL: {url}")
    except Exception as e:
        print(f"\nUpload failed: {e}")
        print("Please verify your AZURE_STORAGE_CONNECTION_STRING in the .env file.")
        
    # Clean up local file
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    main()
