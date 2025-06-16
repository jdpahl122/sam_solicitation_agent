"""Archive solicitation JSON payloads to MinIO S3 storage."""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Iterable, Dict

import boto3
from botocore.exceptions import ClientError

from .base_task import BaseTask


class ArchiveSolicitationsTask(BaseTask):
    """Save full solicitation records to a MinIO bucket and local disk."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        bucket: str = "sam-archive",
        endpoint_url: str = "http://localhost:9000",
        dry_run: bool = False,
    ) -> None:
        self.bucket = bucket
        self.dry_run = dry_run

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
        )

        if not self.dry_run:
            self._ensure_bucket()

    # ------------------------------------------------------------------
    def _ensure_bucket(self) -> None:
        """Create the archive bucket if it does not exist."""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            print(f"📁 Creating bucket '{self.bucket}'")
            self.s3.create_bucket(Bucket=self.bucket)

    # ------------------------------------------------------------------
    def _object_exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    # ------------------------------------------------------------------
    def _upload_with_retry(self, data: Dict, key: str, retries: int = 3) -> bool:
        """Upload JSON data to S3 with retry logic."""
        body = json.dumps(data).encode("utf-8")
        for attempt in range(1, retries + 1):
            try:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=body)
                print(f"✅ Uploaded {key}")
                return True
            except Exception as e:
                print(f"⚠️ Upload failed ({attempt}/{retries}) for {key}: {e}")
        return False

    # ------------------------------------------------------------------
    def _save_local_copy(self, data: Dict, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # ------------------------------------------------------------------
    def execute(self, opportunities: Iterable[Dict]) -> None:
        for record in opportunities:
            notice_id = record.get("noticeId")
            posted = record.get("postedDate")
            if not notice_id or not posted:
                continue

            try:
                dt = datetime.fromisoformat(posted.replace("Z", "+00:00"))
            except Exception:
                # Fallback to simple date parsing
                try:
                    dt = datetime.strptime(posted, "%m/%d/%Y")
                except Exception:
                    print(f"⚠️ Could not parse postedDate '{posted}'")
                    continue

            key_prefix = dt.strftime("%Y/%m/%d")
            key = f"{key_prefix}/{notice_id}.json"

            local_path = os.path.join("raw_data", key)
            self._save_local_copy(record, local_path)

            if self.dry_run:
                print(f"[DRY RUN] Would upload {key}")
                continue

            if self._object_exists(key):
                print(f"⏭️ Skipping existing {key}")
                continue

            self._upload_with_retry(record, key)

