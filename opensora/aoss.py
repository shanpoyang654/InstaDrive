import io
import os

# Install this SDK following the SenseCore document.
# https://console.sensecore.cn/help/docs/cloud-foundation/storage/aoss/#%E6%8E%A8%E8%8D%90%E4%BD%BF%E7%94%A8%E5%B7%A5%E5%85%B7
import aoss_client.client

class AossFile(io.RawIOBase):
    def __init__(self, client, s3_path: str):
        super().__init__()
        self.client = client
        self.s3_path = s3_path
        self.length = self.client.size(s3_path)
        self.p = 0
        self.is_closed = False

    def close(self):
        self.is_closed = True

    @property
    def closed(self):
        return self.is_closed

    def readable(self):
        return True

    def read(self, size=-1):
        read_count = min(size, self.length - self.p) \
            if size >= 0 else self.length - self.p
        if read_count == 0:
            return b""

        data = self.client.get(
            self.s3_path,
            range="{}-{}".format(self.p, self.p + read_count - 1))
        self.p += read_count
        return data

    def readall(self):
        return self.read(-1)

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.p = max(0, min(self.length, offset))
        elif whence == os.SEEK_CUR:
            self.p = max(0, min(self.length, self.p + offset))
        elif whence == os.SEEK_END:
            self.p = max(0, min(self.length, self.length + offset))

        return self.p

    def seekable(self):
        return True

    def tell(self):
        return self.p

    def writable(self):
        return False


class AossLazyFile():
    def __init__(self, client_config_path: str, s3_path: str):
        self.client = aoss_client.client.Client(client_config_path)
        self.s3_path = s3_path

    def open(self, **kwargs):
        return AossFile(self.client, self.s3_path, **kwargs)
