import argparse
import json
import boto3

from utils.env_loader import load_env
from utils.solicitation_assets import enrich_record_with_details


def main(key: str):
    config = load_env()
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id=config.get("MINIO_ACCESS_KEY"),
        aws_secret_access_key=config.get("MINIO_SECRET_KEY"),
        region_name="us-east-1",
    )
    bucket = "sam-archive"
    obj = s3.get_object(Bucket=bucket, Key=key)
    record = json.loads(obj["Body"].read())

    enriched = enrich_record_with_details(
        record, s3, bucket, api_key=config.get("SAM_API_KEY"), dry_run=False
    )

    print(json.dumps(
        {
            "description_key": enriched.get("description_data_key"),
            "attachment_keys": enriched.get("attachment_keys", []),
        },
        indent=2,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch description and attachments for a saved SAM notice"
    )
    parser.add_argument(
        "key", help="S3 key for the JSON file, e.g. 2025/06/16/<notice_id>.json"
    )
    args = parser.parse_args()
    main(args.key)
