"""Minimal, dependency-free NBT (Named Binary Tag) writer and reader.

Minecraft Java stores chunks and level.dat in NBT. We only need a small,
well-tested subset, so we implement it ourselves instead of depending on a
library that may lag behind new Minecraft versions.

Python values are mapped to NBT tags like this:
    Byte(x), Short(x), Int(x), Long(x), Float(x), Double(x)  -> explicit wrappers
    int   -> TAG_Int,  float -> TAG_Double,  bool -> TAG_Byte
    str   -> TAG_String
    dict  -> TAG_Compound (insertion order preserved)
    List(tag_type, items) / list -> TAG_List
    numpy int8 array  -> TAG_Byte_Array
    numpy int32 array -> TAG_Int_Array
    numpy int64 array -> TAG_Long_Array
"""
from __future__ import annotations

import gzip
import io
import struct
from typing import Any

import numpy as np

TAG_END, TAG_BYTE, TAG_SHORT, TAG_INT, TAG_LONG = 0, 1, 2, 3, 4
TAG_FLOAT, TAG_DOUBLE, TAG_BYTE_ARRAY, TAG_STRING = 5, 6, 7, 8
TAG_LIST, TAG_COMPOUND, TAG_INT_ARRAY, TAG_LONG_ARRAY = 9, 10, 11, 12


class _Num:
    """Base class for explicitly typed numbers."""
    tag = -1
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"{type(self).__name__}({self.value!r})"

    def __eq__(self, other):
        return type(other) is type(self) and other.value == self.value


class Byte(_Num):
    tag = TAG_BYTE


class Short(_Num):
    tag = TAG_SHORT


class Int(_Num):
    tag = TAG_INT


class Long(_Num):
    tag = TAG_LONG


class Float(_Num):
    tag = TAG_FLOAT


class Double(_Num):
    tag = TAG_DOUBLE


class List(list):
    """TAG_List with an explicit element type (needed for empty lists)."""

    def __init__(self, elem_tag: int, items=()):
        super().__init__(items)
        self.elem_tag = elem_tag


def _tag_of(v: Any) -> int:
    if isinstance(v, _Num):
        return v.tag
    if isinstance(v, bool):
        return TAG_BYTE
    if isinstance(v, int):
        return TAG_INT
    if isinstance(v, float):
        return TAG_DOUBLE
    if isinstance(v, str):
        return TAG_STRING
    if isinstance(v, dict):
        return TAG_COMPOUND
    if isinstance(v, list):
        return TAG_LIST
    if isinstance(v, np.ndarray):
        if v.dtype == np.int8 or v.dtype == np.uint8:
            return TAG_BYTE_ARRAY
        if v.dtype == np.int32:
            return TAG_INT_ARRAY
        if v.dtype == np.int64 or v.dtype == np.uint64:
            return TAG_LONG_ARRAY
    raise TypeError(f"Cannot encode {type(v)} as NBT")


def _write_payload(out: io.BytesIO, tag: int, v: Any) -> None:
    w = out.write
    if tag == TAG_BYTE:
        w(struct.pack(">b", int(v.value if isinstance(v, _Num) else v)))
    elif tag == TAG_SHORT:
        w(struct.pack(">h", int(v.value)))
    elif tag == TAG_INT:
        w(struct.pack(">i", int(v.value if isinstance(v, _Num) else v)))
    elif tag == TAG_LONG:
        w(struct.pack(">q", int(v.value)))
    elif tag == TAG_FLOAT:
        w(struct.pack(">f", float(v.value)))
    elif tag == TAG_DOUBLE:
        w(struct.pack(">d", float(v.value if isinstance(v, _Num) else v)))
    elif tag == TAG_STRING:
        b = v.encode("utf-8")  # Java uses modified UTF-8; identical for BMP text without NUL
        w(struct.pack(">H", len(b)))
        w(b)
    elif tag == TAG_COMPOUND:
        for k, child in v.items():
            ct = _tag_of(child)
            w(struct.pack(">b", ct))
            kb = k.encode("utf-8")
            w(struct.pack(">H", len(kb)))
            w(kb)
            _write_payload(out, ct, child)
        w(b"\x00")
    elif tag == TAG_LIST:
        if isinstance(v, List):
            et = v.elem_tag
        elif len(v) == 0:
            et = TAG_END
        else:
            et = _tag_of(v[0])
        w(struct.pack(">bi", et, len(v)))
        for item in v:
            _write_payload(out, et, item)
    elif tag == TAG_BYTE_ARRAY:
        w(struct.pack(">i", v.size))
        w(v.astype(np.int8).tobytes())
    elif tag == TAG_INT_ARRAY:
        w(struct.pack(">i", v.size))
        w(v.astype(">i4").tobytes())
    elif tag == TAG_LONG_ARRAY:
        w(struct.pack(">i", v.size))
        w(v.astype(">i8").tobytes())
    else:
        raise ValueError(tag)


def encode(root: dict, name: str = "") -> bytes:
    """Serialize a root compound to uncompressed NBT bytes."""
    out = io.BytesIO()
    out.write(struct.pack(">b", TAG_COMPOUND))
    nb = name.encode("utf-8")
    out.write(struct.pack(">H", len(nb)))
    out.write(nb)
    _write_payload(out, TAG_COMPOUND, root)
    return out.getvalue()


def write_gzip_file(path, root: dict) -> None:
    with gzip.open(path, "wb") as f:
        f.write(encode(root))


# ---------------------------------------------------------------- reader
class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.p = 0

    def take(self, n):
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def unpack(self, fmt):
        s = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(s))[0]

    def string(self):
        n = self.unpack(">H")
        return self.take(n).decode("utf-8", errors="replace")

    def payload(self, tag):
        if tag == TAG_BYTE:
            return Byte(self.unpack(">b"))
        if tag == TAG_SHORT:
            return Short(self.unpack(">h"))
        if tag == TAG_INT:
            return Int(self.unpack(">i"))
        if tag == TAG_LONG:
            return Long(self.unpack(">q"))
        if tag == TAG_FLOAT:
            return Float(self.unpack(">f"))
        if tag == TAG_DOUBLE:
            return Double(self.unpack(">d"))
        if tag == TAG_STRING:
            return self.string()
        if tag == TAG_BYTE_ARRAY:
            n = self.unpack(">i")
            return np.frombuffer(self.take(n), dtype=np.int8).copy()
        if tag == TAG_INT_ARRAY:
            n = self.unpack(">i")
            return np.frombuffer(self.take(4 * n), dtype=">i4").astype(np.int32)
        if tag == TAG_LONG_ARRAY:
            n = self.unpack(">i")
            return np.frombuffer(self.take(8 * n), dtype=">i8").astype(np.int64)
        if tag == TAG_LIST:
            et = self.unpack(">b")
            n = self.unpack(">i")
            return List(et, [self.payload(et) for _ in range(n)])
        if tag == TAG_COMPOUND:
            out = {}
            while True:
                t = self.unpack(">b")
                if t == TAG_END:
                    return out
                k = self.string()
                out[k] = self.payload(t)
        raise ValueError(f"bad tag {tag}")


def decode(data: bytes) -> dict:
    r = _Reader(data)
    t = r.unpack(">b")
    if t != TAG_COMPOUND:
        raise ValueError("root is not a compound")
    r.string()
    return r.payload(TAG_COMPOUND)


def read_gzip_file(path) -> dict:
    with gzip.open(path, "rb") as f:
        return decode(f.read())


def plain(v):
    """Convert decoded NBT into plain Python values (for debugging / tests)."""
    if isinstance(v, _Num):
        return v.value
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
