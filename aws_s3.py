# eğer ihlal tespiti yapılırsa object_url linki mongodb'ye diger bilgilerle birlikte kayıt edilecek.

import boto3
import mimetypes
from pathlib import Path

def upload_to_s3_bucket(video_id, violation_name):
    file_url = Path( "received_output/{}.mp4".format(video_id))
    file_url = str(file_url)
    mimeType, _ = mimetypes.guess_type(
        file_url
        )
    if mimeType is None:
        raise Exception("Failed to quess mimetype")

    s3 = boto3.client("s3")
    bucket_location = s3.get_bucket_location(Bucket="violation-video-bucket")
    object_url = "https://s3-{0}.amazonaws.com/{1}/{2}".format(
        bucket_location['LocationConstraint'],
        "violation-video-bucket",
        "{}/{}.mp4".format(violation_name,video_id))

    s3.upload_file(
        Filename=file_url,
        Bucket="violation-video-bucket",
        Key="{}/{}.mp4".format(violation_name, video_id),
        ExtraArgs={"ContentType": mimeType}

    )
    return object_url
