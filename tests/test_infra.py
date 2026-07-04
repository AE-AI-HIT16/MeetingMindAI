"""Script to test PostgreSQL and MinIO connections."""

import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Import SQLAlchemy components (used by SQLModel under the hood)
from sqlalchemy import create_engine, text

def test_postgres():
    print("\n--- Testing PostgreSQL Connection ---")
    load_dotenv()
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("[FAIL] MinIO credentials not found in .env file.")
        return False
        
    print(f"Attempting to connect to: {db_url}")
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            for row in result:
                if row[0] == 1:
                    print("[OK] Successfully connected to PostgreSQL!")
                    return True
    except Exception as e:
        print(f"[FAIL] Failed to connect to PostgreSQL: {e}")
    return False

def test_minio():
    print("\n--- Testing MinIO (S3) Connection ---")
    load_dotenv()
    minio_user = os.environ.get("MINIO_ROOT_USER", "admin")
    minio_password = os.environ.get("MINIO_ROOT_PASSWORD", "password123")
    
    endpoint_url = "http://localhost:9000"
    print(f"Attempting to connect to MinIO at: {endpoint_url}")
    
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=minio_user,
            aws_secret_access_key=minio_password,
            # Bỏ qua SSL/TLS signature version 4 lỗi của boto3 khi dùng với minio
            region_name="us-east-1"
        )
        
        # Test 1: List buckets
        response = s3.list_buckets()
        print(f"[OK] Successfully connected to MinIO!")
        print(f"   Current buckets: {[bucket['Name'] for bucket in response.get('Buckets', [])]}")
        
        # Test 2: Try creating a test bucket
        bucket_name = "test-bucket"
        try:
            s3.create_bucket(Bucket=bucket_name)
            print(f"[OK] Successfully created a test bucket: '{bucket_name}'")
        except ClientError as e:
            if e.response['Error']['Code'] in ['BucketAlreadyOwnedByYou', 'BucketAlreadyExists']:
                print(f"[OK] Test bucket '{bucket_name}' already exists.")
            else:
                print(f"[WARN] Could not create bucket: {e}")
                
        return True
    except Exception as e:
        print(f"[FAIL] Failed to connect to MinIO: {e}")
        return False

if __name__ == "__main__":
    print("Starting Infrastructure Tests...")
    pg_ok = test_postgres()
    minio_ok = test_minio()
    
    print("\n================ SUMMARY ================")
    if pg_ok and minio_ok:
        print(">>> ALL SYSTEMS GO! PostgreSQL and MinIO are perfectly set up.")
    else:
        print(">>> SOME SYSTEMS FAILED. Please ensure Docker is running and try again.")
