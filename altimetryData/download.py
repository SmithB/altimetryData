"""
Fetch granules from NSIDC (Earthdata Cloud).

Two transports:
  * https (anywhere): Earthdata Login via ~/.netrc (machine urs.earthdata.nasa.gov) or a bearer token in the
    EARTHDATA_TOKEN environment variable;
  * s3 (only from AWS us-west-2, e.g. NASA MAAP): temporary credentials from NSIDC's s3credentials endpoint,
    which last one hour and are refreshed automatically.  Needs boto3.
"""

import os
import shutil
import threading
import time

import requests

S3_CREDENTIALS_URL = "https://data.nsidc.earthdatacloud.nasa.gov/s3credentials"
CHUNK = 8 * 1048576


class HttpsFetcher:
    def __init__(self, token=None, retries=4):
        self.session = requests.Session()
        token = token or os.environ.get("EARTHDATA_TOKEN")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.retries = retries

    def fetch(self, url, out_file):
        tmp = out_file + ".part"
        for attempt in range(self.retries):
            try:
                with self.session.get(url, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in response.iter_content(CHUNK):
                            fh.write(chunk)
                os.replace(tmp, out_file)
                return out_file
            except (requests.RequestException, OSError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)


class S3Fetcher:
    """direct S3 reads from the NSIDC bucket; only works from compute in AWS us-west-2"""

    def __init__(self, refresh_seconds=50 * 60):
        import boto3  # noqa: F401  (fail early if missing)
        self._https = HttpsFetcher()
        self._lock = threading.Lock()
        self._client = None
        self._expires = 0
        self._refresh = refresh_seconds

    def _s3(self):
        with self._lock:
            if self._client is None or time.time() > self._expires:
                import boto3
                creds = self._https.session.get(S3_CREDENTIALS_URL, timeout=60).json()
                self._client = boto3.client("s3", region_name="us-west-2",
                                            aws_access_key_id=creds["accessKeyId"],
                                            aws_secret_access_key=creds["secretAccessKey"],
                                            aws_session_token=creds["sessionToken"])
                self._expires = time.time() + self._refresh
            return self._client

    def fetch(self, url, out_file):
        bucket, key = url.split("s3://", 1)[1].split("/", 1)
        tmp = out_file + ".part"
        self._s3().download_file(bucket, key, tmp)
        os.replace(tmp, out_file)
        return out_file


def fetcher(transport):
    return S3Fetcher() if transport == "s3" else HttpsFetcher()


def fetch_granule(fetch, granule, transport, work_dir):
    """download one granule (a lookup-table row) into its own new directory; returns the local path"""
    url = granule["s3_url"] if transport == "s3" else granule["https_url"]
    out_dir = os.path.join(work_dir, f"{int(granule['granule_id']):08d}")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return fetch.fetch(url, os.path.join(out_dir, os.path.basename(url)))
