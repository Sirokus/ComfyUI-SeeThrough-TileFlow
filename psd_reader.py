"""Minimal 8-bit RGB PSD reader for the tile merge step.

Reads the files written by :mod:`psd_writer` (one layer per part, normal
blend mode, RLE or raw channels) and returns layer names, bounds and RGBA
pixels.  It is deliberately strict about the offsets it consumes so a broken
or foreign PSD fails loudly instead of producing ghost layers.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List

import numpy as np


def unpackbits(data: bytes, expected: int) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)
    while i < n and len(out) < expected:
        h = data[i]
        i += 1
        if h < 128:
            count = h + 1
            out += data[i:i + count]
            i += count
        elif h > 128:
            count = 257 - h
            if i < n:
                out += bytes([data[i]]) * count
                i += 1
        # h == 128: no-op
    if len(out) < expected:
        out += b"\0" * (expected - len(out))
    return bytes(out[:expected])


class _Cursor:
    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def take(self, n: int) -> bytes:
        chunk = self.data[self.pos:self.pos + n]
        if len(chunk) != n:
            raise ValueError("unexpected end of PSD at %d" % self.pos)
        self.pos += n
        return chunk

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack(">H", self.take(2))[0]

    def i16(self) -> int:
        return struct.unpack(">h", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def pascal(self) -> str:
        length = self.u8()
        raw = self.take(length)
        pad = (4 - ((length + 1) % 4)) % 4
        self.take(pad)
        return raw.decode("utf-8", "replace")


def _read_channel(cur: _Cursor, height: int, width: int, compression: int) -> np.ndarray:
    if compression == 0:
        return np.frombuffer(cur.take(height * width), dtype=np.uint8).reshape(height, width).copy()
    if compression != 1:
        raise ValueError("unsupported PSD channel compression %d" % compression)
    lengths = [cur.u16() for _ in range(height)]
    rows = np.empty((height, width), dtype=np.uint8)
    for y, length in enumerate(lengths):
        packed = cur.take(length)
        rows[y] = np.frombuffer(unpackbits(packed, width), dtype=np.uint8)
    return rows


def read_psd(path: str, with_composite: bool = False) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = f.read()
    cur = _Cursor(data)

    if cur.take(4) != b"8BPS":
        raise ValueError("not a PSD file: %s" % path)
    version = cur.u16()
    cur.take(6)
    channels = cur.u16()
    height = cur.u32()
    width = cur.u32()
    depth = cur.u16()
    color_mode = cur.u16()
    if version != 1 or depth != 8 or color_mode != 3:
        raise ValueError("unsupported PSD header: %r" % ((version, depth, color_mode),))

    cur.take(cur.u32())  # colour mode data
    cur.take(cur.u32())  # image resources
    layer_mask_len = cur.u32()
    layer_mask_end = cur.pos + layer_mask_len

    layer_info_len = cur.u32()
    layer_info_end = cur.pos + layer_info_len
    count = abs(cur.i16())

    layers: List[Dict[str, Any]] = []
    for _ in range(count):
        top = cur.i32()
        left = cur.i32()
        bottom = cur.i32()
        right = cur.i32()
        n_channels = cur.u16()
        channel_ids = []
        channel_lengths = []
        for _c in range(n_channels):
            channel_ids.append(cur.i16())
            channel_lengths.append(cur.u32())
        if cur.take(4) != b"8BIM":
            raise ValueError("bad blend signature")
        blend = cur.take(4)
        opacity = cur.u8()
        clipping = cur.u8()
        flags = cur.u8()
        cur.take(1)  # filler
        extra_len = cur.u32()
        extra_end = cur.pos + extra_len
        cur.take(cur.u32())  # layer mask data
        cur.take(cur.u32())  # blending ranges
        name = cur.pascal()
        unicode_name = None
        section_types = []
        while cur.pos < extra_end:
            sig = cur.take(4)
            if sig != b"8BIM":
                raise ValueError("bad additional-layer-info signature %r" % (sig,))
            key = cur.take(4)
            size = cur.u32()
            payload = cur.take(size)
            pad = (4 - (size % 4)) % 4
            if pad and cur.pos + pad <= extra_end:
                cur.take(pad)
            if key == b"luni":
                n = struct.unpack(">I", payload[:4])[0]
                unicode_name = payload[4:4 + n * 2].decode("utf-16-be", "replace")
            elif key == b"lsct" and len(payload) >= 4:
                # Section divider: 1/2 start a folder (open/closed), 3 ends it.
                section_types.append(struct.unpack(">I", payload[:4])[0])
        if 1 in section_types or 2 in section_types:
            kind = "group_open"
        elif 3 in section_types:
            kind = "group_end"
        else:
            kind = "layer"
        layers.append({
            "top": top, "left": left, "bottom": bottom, "right": right,
            "name": unicode_name or name,
            "blend": blend, "opacity": opacity, "clipping": clipping,
            "flags": flags, "hidden": bool(flags & 0x02),
            "kind": kind, "section_types": section_types,
            "channels": list(zip(channel_ids, channel_lengths)),
        })

    for layer in layers:
        h = layer["bottom"] - layer["top"]
        w = layer["right"] - layer["left"]
        planes = {}
        for channel_id, length in layer["channels"]:
            start = cur.pos
            compression = cur.u16()
            plane = _read_channel(cur, h, w, compression)
            if cur.pos - start != length:
                raise ValueError("channel length mismatch for %r" % layer["name"])
            planes[channel_id] = plane
        layer["pixels"] = np.dstack([
            planes.get(0, np.zeros((h, w), np.uint8)),
            planes.get(1, np.zeros((h, w), np.uint8)),
            planes.get(2, np.zeros((h, w), np.uint8)),
            planes.get(-1 if -1 in planes else 3, np.full((h, w), 255, np.uint8)),
        ])

    result = {"width": width, "height": height, "channels": channels, "layers": layers}
    if not with_composite:
        return result

    cur.pos = layer_info_end
    cur.take(cur.u32())
    cur.pos = layer_mask_end

    compression = cur.u16()
    if compression == 1:
        tables = [[cur.u16() for _ in range(height)] for _ in range(4)]
        planes = []
        for table in tables:
            rows = np.empty((height, width), dtype=np.uint8)
            for y, length in enumerate(table):
                packed = cur.take(length)
                rows[y] = np.frombuffer(unpackbits(packed, width), dtype=np.uint8)
            planes.append(rows)
        result["composite"] = np.dstack(planes)
    else:
        result["composite"] = np.dstack([
            _read_channel(cur, height, width, compression) for _ in range(4)
        ])
    return result
