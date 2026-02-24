"""
Robust `.mbox` file parser using Python's ``mailbox`` module.

Extracts subject, sender, date, and body text from each message.
"""

from __future__ import annotations

import email
import mailbox
import tempfile
from email import policy
from pathlib import Path
from typing import IO

import pandas as pd
from bs4 import BeautifulSoup

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_body(msg: email.message.Message) -> str:
    """
    Walk a MIME message and return the best plain-text body.

    Preference: text/plain → text/html (stripped) → empty string.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            # skip attachments
            if part.get("Content-Disposition", "").startswith("attachment"):
                continue
            try:
                payload: str = part.get_payload(decode=True).decode(
                    part.get_content_charset() or "utf-8", errors="replace"
                )
            except Exception:
                continue
            if ct == "text/plain":
                plain_parts.append(payload)
            elif ct == "text/html":
                html_parts.append(payload)
    else:
        try:
            payload = msg.get_payload(decode=True).decode(
                msg.get_content_charset() or "utf-8", errors="replace"
            )
        except Exception:
            payload = ""
        if msg.get_content_type() == "text/html":
            html_parts.append(payload)
        else:
            plain_parts.append(payload)

    if plain_parts:
        return "\n".join(plain_parts)
    if html_parts:
        return BeautifulSoup("\n".join(html_parts), "lxml").get_text(separator=" ")
    return ""


def parse_mbox(source: str | Path | IO[bytes]) -> pd.DataFrame:
    """
    Parse an mbox file and return a DataFrame with columns:
    ``email_index``, ``subject``, ``sender``, ``date``, ``body``.

    Parameters
    ----------
    source
        A file path (str / Path) **or** a file-like bytes object
        (e.g. from ``st.file_uploader``).

    Raises
    ------
    ValueError
        When no messages could be extracted.
    """
    # If source is a file-like object, write to a temp file first
    # (mailbox.mbox requires a real path).
    if hasattr(source, "read"):
        tmp = tempfile.NamedTemporaryFile(suffix=".mbox", delete=False)
        tmp.write(source.read())
        tmp.flush()
        mbox_path = tmp.name
    else:
        mbox_path = str(source)

    logger.info("Parsing mbox: %s", mbox_path)
    mbox = mailbox.mbox(mbox_path)

    records: list[dict[str, str | int]] = []
    for idx, msg in enumerate(mbox):
        try:
            body = _extract_body(msg)
            records.append(
                {
                    "email_index": idx,
                    "subject": msg.get("Subject", ""),
                    "sender": msg.get("From", ""),
                    "date": msg.get("Date", ""),
                    "body": body,
                }
            )
        except Exception as exc:
            logger.warning("Skipping message %d: %s", idx, exc)

    mbox.close()

    if not records:
        raise ValueError("No messages could be extracted from the mbox file.")

    df = pd.DataFrame(records)
    logger.info("Parsed %d emails from mbox.", len(df))
    return df
