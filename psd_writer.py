"""Minimal PSD writer: RGB/8-bit with RGBA layers.

Only what the SeeThrough layer-upscale flow needs is implemented:

* 8 bit per channel, RGB colour mode, 4 channels (R, G, B, A)
* one PSD layer per input layer, normal blend mode, 100 % opacity
* optional PackBits RLE compression (default) or raw channel data
* merged composite image written as RGBA

The structure follows the Adobe Photoshop File Format specification; it is
verified by ``tests/test_offline.py`` (decoded back with a strict reader and
Pillow) and by the ag-psd reader used by the SeeThrough frontend.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image

BLEND_NORMAL = b"norm"


# ---------------------------------------------------------------------------
# PackBits (PSD "RLE")
# ---------------------------------------------------------------------------

def packbits_encode(data: bytes) -> bytes:
    out = bytearray()
    n = len(data)
    i = 0
    while i < n:
        # Find the run length starting at i.
        run = 1
        while i + run < n and data[i + run] == data[i] and run < 128:
            run += 1
        if run >= 3:
            out.append((257 - run) & 0xFF)  # -run + 1 as unsigned byte
            out.append(data[i])
            i += run
        else:
            start = i
            i += run
            while i < n and (i - start) < 128:
                r = 1
                while i + r < n and data[i + r] == data[i] and r < 128:
                    r += 1
                if r >= 3:
                    break
                # A literal run may hold at most 128 bytes.  Without this the
                # counter could reach 129, whose -1 byte (0x80) decodes as a
                # repeat run and desynchronises the whole channel.
                if (i - start) + r > 128:
                    break
                i += r
            count = i - start
            out.append(count - 1)  # literal count - 1 (0..127)
            out.extend(data[start:i])
    return bytes(out)


def _rle_rows(channel: np.ndarray) -> tuple[bytes, bytes]:
    """Return (row length table, PackBits row data) for a 2-D uint8 channel."""
    channel = np.ascontiguousarray(channel, dtype=np.uint8)
    height = channel.shape[0]
    encoded = [packbits_encode(channel[y].tobytes()) for y in range(height)]
    lengths = b"".join(struct.pack(">H", len(e)) for e in encoded)
    return lengths, b"".join(encoded)


def _channel_data(channel: np.ndarray, compression: str) -> bytes:
    """Layer channel data: compression marker + row table + row data."""
    if compression == "raw":
        return struct.pack(">H", 0) + np.ascontiguousarray(channel, dtype=np.uint8).tobytes()
    lengths, rows = _rle_rows(channel)
    return struct.pack(">H", 1) + lengths + rows


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def _pascal_name(name: str) -> bytes:
    raw = str(name).encode("utf-8")
    if len(raw) > 255:
        raw = raw[:255].decode("utf-8", "ignore").encode("utf-8")
    pad = (4 - ((len(raw) + 1) % 4)) % 4
    return bytes([len(raw)]) + raw + b"\0" * pad


def _unicode_name_block(name: str) -> bytes:
    text = str(name).encode("utf-16-be")
    payload = struct.pack(">I", len(text) // 2) + text
    # Additional layer information blocks are 4-byte aligned.  The padding is
    # included in the length field, which is what Photoshop and ag-psd write;
    # readers that expect the length to exclude padding still land correctly
    # because the padded length is already a multiple of 4.
    pad = (4 - (len(payload) % 4)) % 4
    payload += b"\0" * pad
    return b"8BIM" + b"luni" + struct.pack(">I", len(payload)) + payload


def _layer_id_block(layer_id: int) -> bytes:
    return b"8BIM" + b"lyid" + struct.pack(">I", 4) + struct.pack(">I", int(layer_id))


def _section_divider_block(kind: str) -> bytes:
    """``lsct`` block that turns a zero-sized record into a folder marker.

    Photoshop stores a group as two divider records around its children: the
    one *below* the children carries type 3 (bounding section divider, named
    ``</Layer group>``) and the one *above* carries type 1 (open folder) or
    2 (closed folder) plus the folder's own blend mode.  The layout here is
    byte-identical to what ag-psd writes, which is what the SeeThrough
    frontend reads.
    """
    if kind == "start":
        payload = struct.pack(">I", 1) + b"8BIM" + b"pass" + b"\0\0\0\0"
    else:
        payload = struct.pack(">I", 3)
    return b"8BIM" + b"lsct" + struct.pack(">I", len(payload)) + payload


def _layer_record(
    name: str,
    top: int,
    left: int,
    bottom: int,
    right: int,
    channel_lengths: Sequence[int],
    layer_id: int,
    extra_block: bytes = b"",
    flags: int = 0,
) -> bytes:
    rec = bytearray()
    rec += struct.pack(">4i", int(top), int(left), int(bottom), int(right))
    rec += struct.pack(">H", len(channel_lengths))
    for channel_id, length in zip((0, 1, 2, -1), channel_lengths):
        rec += struct.pack(">h", channel_id)
        rec += struct.pack(">I", int(length))

    rec += b"8BIM"
    rec += BLEND_NORMAL
    rec += bytes([255, 0, int(flags), 0])  # opacity, clipping, flags, filler

    extra = bytearray()
    extra += struct.pack(">I", 0)  # layer mask data length
    extra += struct.pack(">I", 0)  # layer blending ranges length
    extra += _pascal_name(name)
    extra += _unicode_name_block(name)
    extra += _layer_id_block(layer_id)
    extra += extra_block

    rec += struct.pack(">I", len(extra))
    rec += extra
    return bytes(rec)


GROUP_START = "group_start"
GROUP_END = "group_end"
_GROUP_END_NAME = "</Layer group>"
_EMPTY_CHANNEL = struct.pack(">H", 0)  # raw compression, zero pixels
# PSD layer record flags: bit 1 (0x02) hides the layer/folder.  The writer
# otherwise leaves the byte at 0 for visible layers; folders carry 24.
HIDDEN_FLAG = 0x02


def _visibility_flags(base: int, hidden: bool) -> int:
    return int(base) | (HIDDEN_FLAG if hidden else 0)


def hidden_record_mask(layers: Sequence[Dict[str, Any]]) -> List[bool]:
    """Which layer records sit inside a hidden folder.

    PSD stores a folder as two divider records: the closing one (``type`` 3,
    ``kind`` ``group_end``) *below* the children and the folder header
    (``group_start`` / ``group_open``) *above* them.  Records run bottom ->
    top, so a folder's hidden flag only appears after its children have
    already gone past.  Pair every closing divider with its header first,
    then mark the whole interval.
    """
    mask = [False] * len(layers)
    pending: List[int] = []
    for index, layer in enumerate(layers):
        kind = layer.get("type") or layer.get("kind")
        if kind in (GROUP_END, "group_end"):
            pending.append(index)
        elif kind in (GROUP_START, "group_open"):
            if pending:
                start = pending.pop()
                if layer.get("hidden"):
                    for k in range(start, index + 1):
                        mask[k] = True
    return mask


def _expand_groups(layers: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten ``{"type": "group", "children": [...]}`` into marker records.

    Layer records are written bottom -> top, so a folder's own records go
    *below* its children (``</Layer group>``) and *above* them (the folder
    name).  Only one nesting level is produced by the merge step, but the
    recursion keeps the writer honest for deeper nesting.

    A hidden folder marks both divider records so readers that look at the
    closing record still see the folder as hidden.
    """
    flat: List[Dict[str, Any]] = []
    for layer in layers:
        if layer.get("type") == "group":
            hidden = bool(layer.get("hidden"))
            flat.append({"type": GROUP_END, "name": _GROUP_END_NAME,
                         "hidden": hidden})
            flat.extend(_expand_groups(layer.get("children") or []))
            flat.append({"type": GROUP_START,
                         "name": str(layer.get("name") or "group"),
                         "hidden": hidden})
        else:
            flat.append(layer)
    return flat


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def _normalise_layers(layers: Iterable[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for layer in layers:
        if layer.get("type") in (GROUP_START, GROUP_END):
            marker = {"type": layer["type"], "name": str(layer.get("name") or "")}
            if layer.get("hidden"):
                marker["hidden"] = True
            normalised.append(marker)
            continue
        pixels = np.asarray(layer["pixels"], dtype=np.uint8)
        if pixels.ndim == 2:
            pixels = np.dstack([pixels, pixels, pixels, np.full(pixels.shape, 255, np.uint8)])
        if pixels.shape[-1] == 3:
            pixels = np.dstack([pixels, np.full(pixels.shape[:2], 255, np.uint8)])
        if pixels.shape[-1] != 4:
            continue
        h, w = pixels.shape[:2]
        left = int(layer.get("left", 0))
        top = int(layer.get("top", 0))
        # Clamp inside the document.
        if left < 0 or top < 0 or left + w > width or top + h > height:
            x0 = max(0, min(width, left))
            y0 = max(0, min(height, top))
            x1 = max(x0, min(width, left + w))
            y1 = max(y0, min(height, top + h))
            pixels = pixels[y0 - top:y1 - top, x0 - left:x1 - left]
            left, top = x0, y0
            h, w = pixels.shape[:2]
        if w <= 0 or h <= 0:
            continue
        item = {
            "name": str(layer.get("name", "layer")),
            "left": left,
            "top": top,
            "pixels": np.ascontiguousarray(pixels),
        }
        if layer.get("hidden"):
            item["hidden"] = True
        normalised.append(item)
    return normalised


def _composite(width: int, height: int, layers: Sequence[Dict[str, Any]]) -> np.ndarray:
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    # Folders are records in this list (bottom -> top), so a hidden folder
    # hides everything between its two markers.  The header comes *after*
    # its children in record order, hence the pair-then-paint helper instead
    # of a simple stack.
    suppressed = hidden_record_mask(layers)
    for index, layer in enumerate(layers):
        kind = layer.get("type")
        if kind in (GROUP_START, GROUP_END):
            continue
        if layer.get("hidden") or suppressed[index]:
            continue
        im = Image.fromarray(layer["pixels"], "RGBA")
        canvas.alpha_composite(im, (layer["left"], layer["top"]))
    return np.array(canvas, dtype=np.uint8)


def write_psd(
    path: str,
    layers: Sequence[Dict[str, Any]],
    width: int,
    height: int,
    compression: str = "rle",
) -> str:
    """Write ``layers`` (bottom -> top order) as a layered PSD file."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("PSD canvas must be positive")
    compression = "raw" if str(compression).lower() == "raw" else "rle"

    layers = _normalise_layers(_expand_groups(layers), width, height)
    if not layers:
        raise ValueError("no layers to write")

    channel_payloads: List[List[bytes]] = []
    for layer in layers:
        if layer.get("type") in (GROUP_START, GROUP_END):
            channel_payloads.append([_EMPTY_CHANNEL] * 4)
            continue
        pixels = layer["pixels"]
        channel_payloads.append([
            _channel_data(pixels[..., 0], compression),
            _channel_data(pixels[..., 1], compression),
            _channel_data(pixels[..., 2], compression),
            _channel_data(pixels[..., 3], compression),
        ])

    records = bytearray()
    channel_blob = bytearray()
    for idx, layer in enumerate(layers):
        lengths = [len(ch) for ch in channel_payloads[idx]]
        if layer.get("type") in (GROUP_START, GROUP_END):
            records += _layer_record(
                layer["name"],
                0, 0, 0, 0,
                lengths,
                layer_id=idx + 1,
                extra_block=_section_divider_block(
                    "start" if layer["type"] == GROUP_START else "end"),
                flags=_visibility_flags(24, bool(layer.get("hidden"))),
            )
        else:
            pixels = layer["pixels"]
            h, w = pixels.shape[:2]
            records += _layer_record(
                layer["name"],
                layer["top"],
                layer["left"],
                layer["top"] + h,
                layer["left"] + w,
                lengths,
                layer_id=idx + 1,
                flags=_visibility_flags(0, bool(layer.get("hidden"))),
            )
        for ch in channel_payloads[idx]:
            channel_blob += ch

    layer_info = struct.pack(">h", len(layers)) + bytes(records) + bytes(channel_blob)
    if len(layer_info) % 2:
        layer_info += b"\0"

    # Layer and mask information section:
    #
    #   uint32  section length (everything that follows, rounded up to 2)
    #     uint32  layer info length
    #       int16 layer count
    #       ... layer records ...
    #       ... channel image data ...
    #     uint32  global layer mask info length
    #
    # ``layer_and_mask`` is exactly the section body, so its length is what
    # the section length field must hold.
    layer_and_mask = (
        struct.pack(">I", len(layer_info))
        + layer_info
        + struct.pack(">I", 0)  # global layer mask info length
    )
    if len(layer_and_mask) % 2:
        layer_and_mask += b"\0"

    composite = _composite(width, height, layers)

    image_data = bytearray()
    image_data += struct.pack(">H", 0 if compression == "raw" else 1)
    if compression == "raw":
        for ch in range(4):
            image_data += composite[..., ch].tobytes()
    else:
        # PSD spec: the merged image stores *all* row byte counts first
        # (rows * channels, in channel order), then all compressed rows.
        # Interleaving table+data per channel makes Photoshop read channel
        # data as a byte count and seek past the end of the file.
        tables = []
        rows = []
        for ch in range(4):
            table, data = _rle_rows(composite[..., ch])
            tables.append(table)
            rows.append(data)
        image_data += b"".join(tables)
        image_data += b"".join(rows)

    header = bytearray()
    header += b"8BPS"
    header += struct.pack(">H", 1)  # version
    header += b"\0" * 6             # reserved
    header += struct.pack(">H", 4)  # channels
    header += struct.pack(">I", height)
    header += struct.pack(">I", width)
    header += struct.pack(">H", 8)  # depth
    header += struct.pack(">H", 3)  # RGB colour mode

    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack(">I", 0))  # colour mode data
        f.write(struct.pack(">I", 0))  # image resources
        f.write(struct.pack(">I", len(layer_and_mask)))
        f.write(layer_and_mask)
        f.write(image_data)
    return path
