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
CHUNK = 1048576


class RateLimiter:
    """
    a bandwidth cap shared by every download in every worker process: each chunk reserves time on a shared
    clock (a token bucket in shared memory) and the caller sleeps until its reservation starts.

    next_time and lock are a multiprocessing.Value("d") and Lock created by the parent.  The cap (MB/s; 0 or less
    means none) is max_MBps, unless rate_file exists: its contents (a number) replace the cap, re-read every
    10 s, so the cap can be changed while a run is going.
    """

    def __init__(self, next_time, lock, max_MBps, rate_file=None):
        self.next_time, self.lock = next_time, lock
        self.default = float(max_MBps or 0)
        self.rate_file = rate_file
        self._rate, self._checked = self.default, 0.0

    def rate(self):
        now = time.monotonic()
        if self.rate_file and now - self._checked > 10:
            self._checked = now
            try:
                with open(self.rate_file) as fh:
                    self._rate = float(fh.read().split()[0])
            except (OSError, ValueError, IndexError):
                self._rate = self.default
        return self._rate

    def consume(self, nbytes):
        rate = self.rate()
        if rate <= 0:
            return
        with self.lock:
            now = time.time()
            # idle time is not banked: after a pause the next chunk starts now, so there is no burst
            start = max(now, self.next_time.value)
            self.next_time.value = start + nbytes / (rate * 1e6)
        if start > now:
            time.sleep(start - now)


class HttpsFetcher:
    def __init__(self, token=None, retries=4):
        self.session = requests.Session()
        token = token or os.environ.get("EARTHDATA_TOKEN")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.retries = retries

    def fetch(self, url, out_file, limiter=None):
        tmp = out_file + ".part"
        for attempt in range(self.retries):
            try:
                with self.session.get(url, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in response.iter_content(CHUNK):
                            if limiter is not None:
                                limiter.consume(len(chunk))
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

    def fetch(self, url, out_file, limiter=None):
        bucket, key = url.split("s3://", 1)[1].split("/", 1)
        tmp = out_file + ".part"
        if limiter is None:
            self._s3().download_file(bucket, key, tmp)
        else:
            body = self._s3().get_object(Bucket=bucket, Key=key)["Body"]
            with open(tmp, "wb") as fh:
                for chunk in body.iter_chunks(CHUNK):
                    limiter.consume(len(chunk))
                    fh.write(chunk)
        os.replace(tmp, out_file)
        return out_file


def fetcher(transport):
    return S3Fetcher() if transport == "s3" else HttpsFetcher()


def fetch_granule(fetch, granule, transport, work_dir, limiter=None):
    """download one granule (a lookup-table row) into its own new directory; returns the local path"""
    url = granule["s3_url"] if transport == "s3" else granule["https_url"]
    out_dir = os.path.join(work_dir, f"{int(granule['granule_id']):08d}")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return fetch.fetch(url, os.path.join(out_dir, os.path.basename(url)), limiter=limiter)
