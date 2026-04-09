#!/usr/bin/env python3
"""
Upload trained model checkpoint to GitHub Release.
Creates a release if it doesn't exist, uploads the .pt file as an asset.
"""
import argparse
import base64
import json
import os
import urllib.request


def get_sha(repo, path, token):
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["sha"]
    except:
        return ""


def create_release(repo, tag, title, notes, token):
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json",
               "Content-Type": "application/json"}
    data = json.dumps({"tag_name": tag, "name": title, "body": notes}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases",
        data=data, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["id"]
    except urllib.error.HTTPError as e:
        if e.code == 422:  # already exists
            # Get existing release ID
            req2 = urllib.request.Request(
                f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
                headers=headers
            )
            with urllib.request.urlopen(req2, timeout=30) as r:
                return json.loads(r.read())["id"]
        raise


def upload_asset(repo, release_id, file_path, token):
    filename = os.path.basename(file_path)
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/octet-stream",
    }
    url = f"https://github.com/junior1p/{repo}/releases/{release_id}/assets?name={filename}"

    with open(file_path, "rb") as f:
        content = f.read()

    req = urllib.request.Request(url, data=content, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        result = json.loads(r.read())
        print(f"  Uploaded: {result.get('browser_download_url', filename)}")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="e.g. junior1p/ESM2-small")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt file")
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v1.0.0")
    parser.add_argument("--title", default="", help="Release title")
    parser.add_argument("--notes", default="", help="Release notes")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""),
                        help="GitHub token (or set GITHUB_TOKEN env var)")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: --token required or GITHUB_TOKEN env var")
        return 1

    print(f"Creating release '{args.tag}' on {args.repo}...")
    release_id = create_release(args.repo, args.tag, args.title or args.tag, args.notes, args.token)

    print(f"Uploading {args.checkpoint}...")
    upload_asset(args.repo, release_id, args.checkpoint, args.token)

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
