import importlib
import io
import os
import struct
import zipfile
import zlib


class PartialReadableRawIO(io.RawIOBase):
    def __init__(
        self, base_io_object: io.RawIOBase, start: int, end: int,
        close_with_this_object: bool = False
    ):
        super().__init__()
        self.base_io_object = base_io_object
        self.p = self.start = start
        self.end = end
        self.close_with_this_object = close_with_this_object
        self.base_io_object.seek(start)

    def close(self):
        if self.close_with_this_object:
            self.base_io_object.close()

    @property
    def closed(self):
        return self.base_io_object.closed if self.close_with_this_object \
            else False

    def readable(self):
        return self.base_io_object.readable()

    def read(self, size=-1):
        read_count = min(size, self.end - self.p) \
            if size >= 0 else self.end - self.p
        data = self.base_io_object.read(read_count)
        self.p += read_count
        return data

    def readall(self):
        return self.read(-1)

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            p = max(0, min(self.end - self.start, offset))
        elif whence == os.SEEK_CUR:
            p = max(
                0, min(self.end - self.start, self.p - self.start + offset))
        elif whence == os.SEEK_END:
            p = max(
                0, min(self.end - self.start, self.end - self.start + offset))

        self.p = self.base_io_object.seek(self.start + p, os.SEEK_SET)
        return self.p

    def seekable(self):
        return self.base_io_object.seekable()

    def tell(self):
        return self.p - self.start

    def writable(self):
        return False


class LazyFile():
    def __init__(self, path: str, mode: str = "rb"):
        self.path = path
        self.mode = mode

    def open(self, **kwargs):
        return open(self.path, self.mode, **kwargs)


class StatelessZipFile():
    def __init__(self, lazy_file):
        self.lazy_file = lazy_file
        with self.lazy_file.open() as f:
            with zipfile.ZipFile(f) as zf:
                self.items = {
                    i.filename: i.header_offset
                    for i in zf.infolist()
                }

    def namelist(self):
        return list(self.items.keys())

    def read(self, name: str):
        header_offset = self.items[name]
        with self.lazy_file.open() as f:
            f.seek(header_offset)
            fh = struct.unpack(zipfile.structFileHeader, f.read(30))
            offset = header_offset + 30 + fh[zipfile._FH_FILENAME_LENGTH] + \
                fh[zipfile._FH_EXTRA_FIELD_LENGTH]
            size = fh[zipfile._FH_COMPRESSED_SIZE]
            method = fh[zipfile._FH_COMPRESSION_METHOD]

            f.seek(offset)
            data = f.read(size)

        if method == zipfile.ZIP_STORED:
            return data
        elif method == zipfile.ZIP_DEFLATED:
            return zlib.decompress(data, -15)
        else:
            raise NotImplementedError(
                "That compression method is not supported")

    def get_io_object(self, name: str):
        header_offset = self.items[name]
        f = self.lazy_file.open()
        f.seek(header_offset)
        fh = struct.unpack(zipfile.structFileHeader, f.read(30))
        method = fh[zipfile._FH_COMPRESSION_METHOD]
        assert method == zipfile.ZIP_STORED

        offset = header_offset + 30 + fh[zipfile._FH_FILENAME_LENGTH] + \
            fh[zipfile._FH_EXTRA_FIELD_LENGTH]
        size = fh[zipfile._FH_COMPRESSED_SIZE]
        return PartialReadableRawIO(f, offset, offset + size, True)


class ChainedReaders():
    def __init__(self, reader_list: list):
        self.reader_list = reader_list
        self.dict = {
            j: i_id
            for i_id, i in enumerate(reader_list)
            for j in i.namelist()
        }

    def namelist(self):
        return list([
            j for i in self.reader_list
            for j in i.namelist()
        ])

    def read(self, name: str):
        reader_id = self.dict[name]
        return self.reader_list[reader_id].read(name)

    def get_io_object(self, name: str):
        reader_id = self.dict[name]
        return self.reader_list[reader_id].get_io_object(name)


def create_instance(class_name: str, **kwargs):
    if "." in class_name:
        i = class_name.rfind(".")
        module_name = class_name[:i]
        class_name = class_name[i+1:]

        module_type = importlib.import_module(module_name, package=None)
        class_type = getattr(module_type, class_name)
    elif class_name in globals():
        class_type = globals()[class_name]
    else:
        raise RuntimeError("Failed to find the class {}.".format(class_name))

    return class_type(**kwargs)


def create_instance_from_config(config, level=0, **kwargs):
    if isinstance(config, dict):
        if "_class_name" in config:
            args = {
                k: create_instance_from_config(v, level + 1)
                for k, v in config.items() if k != "_class_name"
            }
            if level == 0:
                args.update(kwargs)

            return create_instance(config["_class_name"], **args)

        else:
            return config

    elif isinstance(config, list):
        return [create_instance_from_config(i, level + 1) for i in config]
    else:
        return config
