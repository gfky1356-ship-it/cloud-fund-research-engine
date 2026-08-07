#!/usr/bin/env python3
"""Upload latest fund research outputs to Google Drive using a service account."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import tempfile
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


SCOPES = ["https://www.googleapis.com/auth/drive"]


def load_credentials():
    secret_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    secret_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if secret_json:
        info = json.loads(secret_json)
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    if secret_path:
        return service_account.Credentials.from_service_account_file(secret_path, scopes=SCOPES)
    raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS")


def find_existing(service, folder_id: str, name: str) -> str | None:
    safe_name = name.replace("'", "\\'")
    query = f"name = '{safe_name}' and '{folder_id}' in parents and trashed = false"
    result = service.files().list(q=query, fields="files(id,name)", pageSize=10).execute()
    files = result.get("files", [])
    return files[0]["id"] if files else None


def upload_one(service, folder_id: str, path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    media = MediaFileUpload(str(path), mimetype=mime_type, resumable=False)
    existing_id = find_existing(service, folder_id, path.name)
    if existing_id:
        updated = (
            service.files()
            .update(fileId=existing_id, media_body=media, fields="id,name,webViewLink")
            .execute()
        )
        return updated.get("webViewLink", existing_id)
    created = (
        service.files()
        .create(
            body={"name": path.name, "parents": [folder_id]},
            media_body=media,
            fields="id,name,webViewLink",
        )
        .execute()
    )
    return created.get("webViewLink", created["id"])


def update_one(service, file_id: str, path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    media = MediaFileUpload(str(path), mimetype=mime_type, resumable=False)
    updated = (
        service.files()
        .update(fileId=file_id, media_body=media, fields="id,name,webViewLink")
        .execute()
    )
    return updated.get("webViewLink", updated["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload fund outputs to Google Drive")
    parser.add_argument("--folder-id", default=os.environ.get("GOOGLE_DRIVE_OUTPUT_FOLDER_ID"))
    parser.add_argument("--csv-file-id", default=os.environ.get("GOOGLE_DRIVE_CSV_FILE_ID"))
    parser.add_argument("--json-file-id", default=os.environ.get("GOOGLE_DRIVE_JSON_FILE_ID"))
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    credentials = load_credentials()
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    for raw in args.files:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        file_id = None
        if path.suffix.lower() == ".csv":
            file_id = args.csv_file_id
        elif path.suffix.lower() == ".json":
            file_id = args.json_file_id
        if file_id:
            link = update_one(service, file_id, path)
        else:
            if not args.folder_id:
                raise RuntimeError("Missing target file id and missing --folder-id or GOOGLE_DRIVE_OUTPUT_FOLDER_ID")
            link = upload_one(service, args.folder_id, path)
        print(f"uploaded {path.name}: {link}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
