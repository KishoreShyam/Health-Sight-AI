import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Initialize BlobServiceClient if connection string is configured
if connection_string:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    except Exception as e:
        print(f"Error initializing BlobServiceClient: {e}")
        blob_service_client = None
else:
    blob_service_client = None

def upload_blob(file_data, file_name, container_name=None):
    """
    Uploads a file (bytes, file-like object, or file path) to Azure Blob Storage.
    Returns the public URL of the uploaded blob.
    """
    if not blob_service_client:
        raise ValueError("Azure Storage connection string is not configured. Please check your .env file.")
    
    if container_name is None:
        container_name = os.getenv("AZURE_STORAGE_CONTAINER", "health-sight-uploads")
    
    # Get or create container
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
        # Set public access policy if needed, default is private
    except ResourceExistsError:
        pass  # Container already exists
    except Exception as e:
        # If permission issues, just attempt to proceed
        print(f"Warning: Container creation/verification failed: {e}")
        
    # Get blob client and upload
    blob_client = container_client.get_blob_client(file_name)
    
    # Handle different input data types
    if isinstance(file_data, str) and os.path.exists(file_data):
        with open(file_data, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)
    else:
        blob_client.upload_blob(file_data, overwrite=True)
        
    return blob_client.url
