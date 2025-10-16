import boto3
from botocore.exceptions import NoCredentialsError
import os
from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REGION = os.getenv("REGION")
# AWS credentials (Use environment variables in production)
# AWS_ACCESS_KEY ="AKIA3DGULCSWXFHUEBJS"
# AWS_SECRET_KEY = "i2/lT+KHHWINDLeQgUgQUkaUNxXzxK1wr2IQ8noJ"
# BUCKET_NAME = 'optifo-dev'
# REGION = 'eu-west-2'  # Example: Mumbai region

# Create S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)

print("S3 client created successfully")




def upload_video_to_s3(file_path, s3_folder="ai_videos"):
    try:
        file_name = os.path.basename(file_path)
        s3_key = f"{s3_folder}/{file_name}"
        # Upload the file
        s3.upload_file(file_path, BUCKET_NAME, s3_key, ExtraArgs={'ACL': 'public-read'})
        # Construct the file URL
        file_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{s3_key}"
        return file_url
    except FileNotFoundError:
        return "File not found"
    except NoCredentialsError:
        return "Credentials not available"


# video_path = "video_bg.mp4"
# video_url= upload_video_to_s3(video_path)

# print(video_url)
