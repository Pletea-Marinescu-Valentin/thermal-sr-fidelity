import io
import urllib.request
import zipfile


class HttpFile(io.RawIOBase):

    def __init__(self, url):
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.final_url = r.geturl()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        elif whence == 2:
            self.pos = self.size + off
        return self.pos

    def read(self, n=-1):
        if n == -1:
            n = self.size - self.pos
        if n == 0:
            return b""
        end = min(self.pos + n - 1, self.size - 1)
        req = urllib.request.Request(
            self.final_url, headers={"Range": f"bytes={self.pos}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(url):
    hf = HttpFile(url)
    return hf, zipfile.ZipFile(io.BufferedReader(hf, buffer_size=1 << 20))
