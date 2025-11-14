import boto3, uuid
from botocore.client import Config
from app.utils.config import settings

_s3 = None

def s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
            config=Config(signature_version="s3v4"),
        )
    return _s3

def ensure_bucket():
    s3 = s3_client()
    buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if settings.S3_BUCKET not in buckets:
        s3.create_bucket(Bucket=settings.S3_BUCKET)

def upload_to_s3(file_bytes: bytes, filename: str) -> str:
    s3 = s3_client()
    ensure_bucket()
    key = f"{uuid.uuid4()}/{filename}"
    s3.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=file_bytes)
    return key

def get_presigned_url(key: str, expires=3600) -> str:
    s3 = s3_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_BUCKET, "Key": key},
        ExpiresIn=expires,
    )
