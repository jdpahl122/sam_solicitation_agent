import os
import json
import io
from typing import List

import boto3
import pdfplumber
from botocore.exceptions import ClientError

from utils.env_loader import load_env
from utils.rag_helpers import filter_valid_opportunities
from rag.faiss_store import FaissStore


def extract_pdf_text(data: bytes) -> str:
    """Return text from a PDF byte string."""
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except Exception as e:
        print(f"⚠️ PDF parse error: {e}")
        return ""


def run() -> None:
    config = load_env()
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id=config.get("MINIO_ACCESS_KEY"),
        aws_secret_access_key=config.get("MINIO_SECRET_KEY"),
        region_name="us-east-1",
    )

    bucket = "sam-archive"
    paginator = s3.get_paginator("list_objects_v2")

    docs: List[dict] = []
    processed = skipped = failed = 0
    pdf_missing = 0

    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            parts = key.split("/")
            # Expect keys like YYYY/MM/DD/<notice_id>.json
            if len(parts) != 4:
                continue

            processed += 1

            try: 
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                record = json.loads(body)
            except Exception as e:
                print(f"⚠️ Failed to load {key}: {e}")
                failed += 1
                continue

            if not filter_valid_opportunities([record]):
                print(f"skipped filter_valid_opportunities {record}")
                skipped += 1
                continue

            set_aside = (record.get("typeOfSetAside") or "").lower()
            if "sba" not in set_aside and "sdvosbc" not in set_aside:
                print(f"skipped set-aside {set_aside}")
                skipped += 1
                continue

            notice_id = record.get("noticeId") or os.path.splitext(os.path.basename(key))[0]
            date_prefix = os.path.dirname(key)
            asset_prefix = f"{date_prefix}/{notice_id}/"

            pdf_texts = []

            # description.json contains additional text if present
            try:
                desc_obj = s3.get_object(Bucket=bucket, Key=f"{asset_prefix}description.json")
                desc_data = json.loads(desc_obj["Body"].read())
                extra_desc = desc_data.get("description") if isinstance(desc_data, dict) else None
            except ClientError:
                extra_desc = None
            except Exception as e:
                print(f"⚠️ Failed to load description for {notice_id}: {e}")
                extra_desc = None

            resp = s3.list_objects_v2(Bucket=bucket, Prefix=asset_prefix)
            if resp.get("KeyCount", 0) == 0:
                pdf_missing += 1
            for item in resp.get("Contents", []):
                if not item["Key"].lower().endswith(".pdf"):
                    continue
                try:
                    pdf_bytes = s3.get_object(Bucket=bucket, Key=item["Key"])["Body"].read()
                    pdf_texts.append(extract_pdf_text(pdf_bytes))
                except Exception as e:
                    print(f"⚠️ Failed to fetch PDF {item['Key']}: {e}")

            text_parts = [record.get("title"), record.get("description")]
            if extra_desc:
                if isinstance(extra_desc, str):
                    text_parts.append(extra_desc)
                else:
                    text_parts.append(json.dumps(extra_desc))
            text_parts.extend(pdf_texts)

            text_blob = "\n\n".join(part for part in text_parts if part)

            metadata = {
                "notice_id": notice_id,
                "title": record.get("title"),
                "naics": record.get("naics") or record.get("naicsCode"),
                "agency": record.get("agency"),
                "setaside": record.get("setAsideCode"),
                "response_deadline": record.get("responseDeadLine") or record.get("responseDeadline"),
                "notice_type": record.get("noticeType"),
                "link": record.get("uiLink") or record.get("url"),
            }

            docs.append({"text": text_blob, "metadata": metadata})
            print(f"✅ Ingested {notice_id}")

    print(
        f"\n📊 Processed: {processed} | Stored: {len(docs)} | "
        f"Skipped: {skipped} | PDFs missing: {pdf_missing} | Failures: {failed}"
    )

    if docs:
        store = FaissStore()
        store.overwrite_documents(docs)
    else:
        print("❌ No documents to store, skipping FAISS index update.")


if __name__ == "__main__":
    run()
