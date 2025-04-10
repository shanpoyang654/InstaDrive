# File system configurations

This project uses the [fsspec](https://github.com/fsspec/filesystem_spec) to access the data from varied sources including the local file system, S3, Zip content.


## Common usage

You can open, read, seek the file with the file system objects. Check out the [usage](https://filesystem-spec.readthedocs.io/en/latest/usage.html) document to quick start.

### Create the file system

``` Python
import fsspec.implementations.local
fs = fsspec.implementations.local.LocalFileSystem()
```

### Open and read the file

``` Python
import json
config_path = "configs/fs/nuscenes_st_aoss.json"
with fs.open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

from PIL import Image
image_path = "samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
with fs.open(image_path, "rb") as f:
    image = Image.open(f)
```


## API reference

### dwm.fs.czip.CombinedZipFileSystem

This file system opens several ZIP blobs from a given file system and provide the file access inside the ZIP blobs. It is forkable and compatible with multi worker data loader of the PyTorch.

### dwm.fs.s3fs.ForkableS3FileSystem

This file system opens the S3 service and provide the file access on the service. It is forkable and compatible with multi worker data loader of the PyTorch.


## Configuration samples
It is easy to initialize the file system object by `dwm.common.create_instance_from_config()` with following configurations by JSON.

### Local file system

``` JSON
{
    "_class_name": "fsspec.implementations.local.LocalFileSystem"
}
```

**Relative directory on local file system**
``` JSON
{
    "_class_name": "fsspec.implementations.dirfs.DirFileSystem",
    "path": "/mnt/storage/user/wuzehuan/Downloads/data/nuscenes",
    "fs": {
        "_class_name": "fsspec.implementations.local.LocalFileSystem"
    }
}
```

### S3 file system

The parameters follow the [Botocore confiruation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html).

``` JSON
{
    "_class_name": "dwm.fs.s3fs.ForkableS3FileSystem",
    "endpoint_url": "http://aoss-internal.st-sh-01.sensecoreapi-oss.cn",
    "aws_access_key_id": "4853705CDE93446E8D902F70291C2C92",
    "aws_secret_access_key": "163175A2193C4B7D80FDE3D1F19804FA"
}
```

**Relative directory on S3 file system**

``` JSON
{
    "_class_name": "fsspec.implementations.dirfs.DirFileSystem",
    "path": "users/wuzehuan/data/nuscenes",
    "fs": {
        "_class_name": "dwm.fs.s3fs.ForkableS3FileSystem",
        "endpoint_url": "http://aoss-internal.st-sh-01.sensecoreapi-oss.cn",
        "aws_access_key_id": "4853705CDE93446E8D902F70291C2C92",
        "aws_secret_access_key": "163175A2193C4B7D80FDE3D1F19804FA"
    }
}
```

**Retry options on S3 file system**

``` JSON
{
    "_class_name": "dwm.fs.s3fs.ForkableS3FileSystem",
    "endpoint_url": "http://aoss-internal.st-sh-01.sensecoreapi-oss.cn",
    "aws_access_key_id": "4853705CDE93446E8D902F70291C2C92",
    "aws_secret_access_key": "163175A2193C4B7D80FDE3D1F19804FA",
    "config": {
        "_class_name": "botocore.config.Config",
        "retries": {
            "max_attempts": 8
        }
    }
}
```


## Project related information

| Cluster | S3 endpoint | AK | Key owner |
| - | - | - | - |
| [cn-sh](http://console.sensecore.cn) | http://aoss.cn-sh-01.sensecoreapi-oss.cn | AE939C3A07AE4E6D93908AA603B9F3A9 | Tian Hao |
| [cn-fj](https://console.cn-fj-01.thinkheadbd.com/) | http://aoss.cn-fj-01.thinkheadbdapi-oss.com | F12314409BA6429F8784BBBEAF1C3E34 | Fan Jianan |
| [st-sh](http://console.sensecore.cn) | http://aoss-v2.st-sh-01.sensecoreapi-oss.cn | 4853705CDE93446E8D902F70291C2C92 | Wu Zehuan |

* Replace the `aoss` with `aoss-internal` for better performance inside the cluster
* Either ask the secret key from the key owner, or ask for the bucket access to your key.
