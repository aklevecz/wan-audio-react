import boto3
from botocore.client import Config
import os
from pathlib import Path

# Load environment variables from .env file when running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system environment variables

# --- Configuration ---
# These are read from environment variables. In Runpod, they should be set as secrets.
# Runpod automatically prefixes secrets with "RUNPOD_SECRET_".
def get_env_vars():
    """Get environment variables dynamically to ensure they're loaded after dotenv."""
    return {
        'ENDPOINT_URL': os.environ.get("R2_ENDPOINT_URL"),
        'ACCESS_KEY_ID': os.environ.get("R2_ACCESS_KEY_ID"),
        'SECRET_ACCESS_KEY': os.environ.get("R2_SECRET_ACCESS_KEY"),
        'BUCKET_NAME': os.environ.get("R2_BUCKET_NAME"),
        'PUBLIC_URL_BASE': os.environ.get("R2_PUBLIC_URL_BASE")
    }

def get_s3_client():
    """Initializes and returns a boto3 S3 client configured for R2."""
    env_vars = get_env_vars()
    if not all([env_vars['ENDPOINT_URL'], env_vars['ACCESS_KEY_ID'], env_vars['SECRET_ACCESS_KEY'], env_vars['BUCKET_NAME']]):
        raise ValueError("One or more R2 environment variables are not set.")

    s3_client = boto3.client(
        's3',
        endpoint_url=env_vars['ENDPOINT_URL'],
        aws_access_key_id=env_vars['ACCESS_KEY_ID'],
        aws_secret_access_key=env_vars['SECRET_ACCESS_KEY'],
        config=Config(signature_version='s3v4'),
        region_name='auto' # R2 specific
    )
    return s3_client

def upload_file(local_path: Path, object_name: str) -> bool:
    """
    Uploads a file to the R2 bucket.

    Args:
        local_path: The path to the local file to upload.
        object_name: The desired key (path) for the object in the bucket.

    Returns:
        True if upload was successful, False otherwise.
    """
    env_vars = get_env_vars()
    print(f"Attempting to upload '{local_path}' to bucket '{env_vars['BUCKET_NAME']}' as '{object_name}'...")
    try:
        s3_client = get_s3_client()
        s3_client.upload_file(str(local_path), env_vars['BUCKET_NAME'], object_name)
        print(f"Successfully uploaded to {object_name}")
        return True
    except Exception as e:
        print(f"Error uploading file: {e}")
        return False

def get_public_url(object_name: str) -> str:
    """
    Constructs the public URL for an object in the R2 bucket.
    
    Note: This assumes your R2 bucket is configured for public access.

    Args:
        object_name: The key (path) of the object in the bucket.

    Returns:
        The full public URL to the object.
    """
    env_vars = get_env_vars()
    if not env_vars['PUBLIC_URL_BASE']:
        raise ValueError("R2_PUBLIC_URL_BASE environment variable is not set.")
        
    return f"{env_vars['PUBLIC_URL_BASE']}/{object_name}"

# --- Example Usage (for testing) ---
def main_test():
    """A simple test function to demonstrate usage."""
    print("--- Running Cloud Storage Test ---")
    
    # Check if environment variables are set
    if not all([os.environ.get(var) for var in ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_PUBLIC_URL_BASE"]]):
        print("\nWARNING: R2 environment variables not set. Skipping test.")
        print("Please set the following environment variables to run this test:")
        print(" - R2_ENDPOINT_URL")
        print(" - R2_ACCESS_KEY_ID")
        print(" - R2_SECRET_ACCESS_KEY")
        print(" - R2_BUCKET_NAME")
        print(" - R2_PUBLIC_URL_BASE")
        return

    # Create a dummy file for testing
    test_dir = Path("./test_upload_dir")
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test.txt"
    with open(test_file, "w") as f:
        f.write("This is a test file for cloud storage upload.")
    
    # Define the object name/key in the bucket
    test_object_name = "test_uploads/my_test_file.txt"

    # Upload the file
    success = upload_file(test_file, test_object_name)

    if success:
        # Get the public URL
        public_url = get_public_url(test_object_name)
        print(f"\nTest successful!")
        print(f"Public URL: {public_url}")
    else:
        print("\nTest failed.")

if __name__ == "__main__":
    main_test()
