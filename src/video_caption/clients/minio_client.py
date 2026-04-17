from pathlib import Path
from minio import Minio


class MinIOClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    def download(self, bucket: str, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._client.fget_object(bucket, key, str(dest))

    def upload(self, bucket: str, key: str, src: Path) -> None:
        self._client.fput_object(bucket, key, str(src))

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        return [obj.object_name for obj in self._client.list_objects(bucket, prefix=prefix)]
