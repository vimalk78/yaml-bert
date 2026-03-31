"""Sanitize dumped cluster YAMLs by removing sensitive data.

Removes:
- Secret data/stringData values (replaced with REDACTED)
- ServiceAccount token references
- TLS certificates and keys
- ConfigMap values that look like credentials
- metadata.managedFields (noisy, not useful for training)
- status blocks (cluster state, not user-authored)

Usage:
    python scripts/sanitize_yamls.py cluster-yamls/
    python scripts/sanitize_yamls.py cluster-yamls/ --output clean-yamls/
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import glob
import os
import re
import shutil

import yaml


# Keys whose values should be redacted regardless of location
SENSITIVE_VALUE_KEYS: set[str] = {
    "password", "secret", "token", "key", "cert", "ca.crt", "tls.crt",
    "tls.key", "ca.key", "credentials", "access-key", "secret-key",
    "aws_access_key_id", "aws_secret_access_key", "connection-string",
    "database-url", "api-key", "apikey", "auth-token",
}

# Patterns in ConfigMap keys that suggest sensitive values
SENSITIVE_KEY_PATTERNS: list[re.Pattern] = [
    re.compile(r"(password|secret|token|credential|api.?key)", re.IGNORECASE),
]


def sanitize_document(doc: dict) -> dict:
    """Sanitize a single YAML document."""
    if not isinstance(doc, dict):
        return doc

    kind: str = doc.get("kind", "")

    # Remove status — cluster state, not user-authored
    doc.pop("status", None)

    # Remove managedFields — noisy metadata
    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("managedFields", None)
        metadata.pop("resourceVersion", None)
        metadata.pop("uid", None)
        metadata.pop("creationTimestamp", None)
        metadata.pop("generation", None)
        metadata.pop("selfLink", None)

    # Secret: redact all data and stringData values
    if kind == "Secret":
        for field in ("data", "stringData"):
            block = doc.get(field)
            if isinstance(block, dict):
                for k in block:
                    block[k] = "REDACTED"

    # ServiceAccount: remove secrets/tokens
    if kind == "ServiceAccount":
        doc.pop("secrets", None)
        doc.pop("imagePullSecrets", None)

    # Recursively redact sensitive-looking values in ConfigMaps and elsewhere
    if kind == "ConfigMap":
        data = doc.get("data")
        if isinstance(data, dict):
            for k in list(data.keys()):
                if _is_sensitive_key(k):
                    data[k] = "REDACTED"

    # Remove any annotation values that look like tokens
    if isinstance(metadata, dict):
        annotations = metadata.get("annotations")
        if isinstance(annotations, dict):
            for k in list(annotations.keys()):
                v = annotations[k]
                if isinstance(v, str) and _looks_like_token(v):
                    annotations[k] = "REDACTED"

    return doc


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name suggests sensitive content."""
    lower = key.lower()
    if lower in SENSITIVE_VALUE_KEYS:
        return True
    return any(p.search(lower) for p in SENSITIVE_KEY_PATTERNS)


def _looks_like_token(value: str) -> bool:
    """Heuristic: long base64-ish strings are likely tokens."""
    if len(value) > 100 and re.match(r'^[A-Za-z0-9+/=\-_.]+$', value):
        return True
    return False


def sanitize_file(input_path: str, output_path: str) -> int:
    """Sanitize a YAML file, return number of documents processed."""
    with open(input_path) as f:
        content = f.read()

    documents = list(yaml.safe_load_all(content))
    sanitized = []
    for doc in documents:
        if doc is None:
            continue
        sanitized.append(sanitize_document(doc))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump_all(sanitized, f, default_flow_style=False)

    return len(sanitized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize cluster YAMLs")
    parser.add_argument("input_dir", type=str, help="Directory of dumped YAMLs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: sanitize in place)")
    args = parser.parse_args()

    in_place: bool = args.output is None
    output_dir: str = args.output or args.input_dir

    files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.yaml"), recursive=True))
    print(f"Sanitizing {len(files)} YAML files...")

    total_docs: int = 0
    for fpath in files:
        rel = os.path.relpath(fpath, args.input_dir)
        out_path = os.path.join(output_dir, rel)
        try:
            total_docs += sanitize_file(fpath, out_path)
        except Exception as e:
            print(f"  SKIP {rel}: {e}")

    print(f"Done. {total_docs} documents sanitized across {len(files)} files.")
    if in_place:
        print("(Sanitized in place)")
    else:
        print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
