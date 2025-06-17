import os
import json
import re
import requests
from typing import Dict, List, Optional, Tuple


def enrich_record_with_details(
    record: Dict,
    s3_client,
    bucket: str,
    *,
    api_key: Optional[str] = None,
    endpoint_url: str = "http://localhost:9000",
    dry_run: bool = False,
) -> Dict:
    """Fetch full description and attachments for a solicitation record.

    If ``record['description']`` is a URL it is fetched and the resulting JSON
    or text is stored in the same S3 prefix as the original record under
    ``<prefix>/<notice_id>/description.json``.

    Any URLs in ``record['resourceLinks']`` are downloaded and uploaded to the
    bucket under ``<prefix>/<notice_id>/``. The resulting S3 keys are stored
    in ``record['attachment_keys']``.
    """
    notice_id = record.get("noticeId")
    posted = record.get("postedDate")
    if not notice_id or not posted:
        return record

    # Compute base prefix (same logic as ArchiveSolicitationsTask)
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(posted.replace("Z", "+00:00"))
    except Exception:
        try:
            from datetime import datetime

            dt = datetime.strptime(posted, "%m/%d/%Y")
        except Exception:
            return record

    key_prefix = dt.strftime("%Y/%m/%d")
    base_prefix = f"{key_prefix}/{notice_id}"

    # ------------------------------------------------------------------
    desc = record.get("description")
    if isinstance(desc, str) and desc.startswith("http"):
        url = desc
        if api_key and "api_key=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}api_key={api_key}"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            try:
                desc_data = resp.json()
            except ValueError:
                desc_data = {"description": resp.text}
            if not dry_run:
                key = f"{base_prefix}/description.json"
                body = json.dumps(desc_data).encode("utf-8")
                s3_client.put_object(Bucket=bucket, Key=key, Body=body)
                record["description_data_key"] = key
            else:
                record["description_data"] = desc_data
        except Exception as e:
            print(f"⚠️ Failed to fetch description for {notice_id}: {e}")

    # ------------------------------------------------------------------
    attachment_keys: List[str] = []
    for link in record.get("resourceLinks", []):
        if not isinstance(link, str) or not link.startswith("http"):
            continue
        try:
            resp = requests.get(link)
            resp.raise_for_status()
            file_id = link.rstrip("/").split("/")[-2]
            filename = file_id
            cd = resp.headers.get("Content-Disposition")
            if cd:
                m = re.search(r"filename=\"?([^\";]+)\"?", cd)
                if m:
                    filename = m.group(1)
            key = f"{base_prefix}/{filename}"
            if not dry_run:
                s3_client.put_object(Bucket=bucket, Key=key, Body=resp.content)
            attachment_keys.append(key)
        except Exception as e:
            print(f"⚠️ Failed to download attachment {link}: {e}")

    if attachment_keys:
        record["attachment_keys"] = attachment_keys

    return record


def parse_s3_path(path: str, default_bucket: str = "sam-archive") -> Tuple[str, str]:
    """Split an S3 path into bucket and key.

    Parameters
    ----------
    path : str
        Either ``<bucket>/<key>`` or just ``<key>``.  If only a key is
        provided, ``default_bucket`` is returned as the bucket name.

    Returns
    -------
    Tuple[str, str]
        The bucket and key values.
    """
    if "/" not in path:
        return default_bucket, path

    first, rest = path.split("/", 1)
    if first.isdigit():
        # Looks like a year prefix rather than a bucket name
        return default_bucket, path
    return first, rest
