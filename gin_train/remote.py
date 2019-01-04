"""Test fs
"""
from fs.osfs import OSFS
from fs.copy import copy_file_if_newer
from fs import open_fs
import os
from tqdm import tqdm


def _open_fs(directory):
    if directory.startswith("s3://"):
        """Manually fetch the permissions from the environment

        Requires the following env variables:
        - S3_ACCESS_KEY
        - S3_SECRET_KEY
        - S3_URL
        """
        from fs_s3fs import S3FS
        if not directory.endswith("/"):
            directory += "/"

        bucket, fpath = directory[len("s3://"):].split("/", 1)
        return S3FS(bucket, dir_path=fpath,
                    aws_access_key_id=os.environ.get("S3_ACCESS_KEY", None),
                    aws_secret_access_key=os.environ.get('S3_SECRET_KEY', None),
                    strict=False,
                    endpoint_url=os.environ.get('S3_URL', None))
    else:
        return open_fs(directory)


def upload_dir(local_dir, remote_dir):
    """Upload directory to the remote directory

    Args:
      local_dir: local directory
      remote_dir: remote directory

    Returns:
      List of urls
    """
    remote_fs = _open_fs(remote_dir)
    local_fs = OSFS(local_dir)
    # urls = []
    for f in tqdm(list(local_fs.walk.files())):
        copy_file_if_newer(local_fs, f, remote_fs, f)
        # urls.append((f, remote_fs.geturl(f)))
    # return urls


# upload_dir("tests", 's3://bucket1/asdas/folder2')
