"""CPU-only, bounded safetensors metadata checks for the DeepSeek-V4.1 engram tables.

The engram tables are never loaded as tensors: the V4.1 engram module gathers
their rows straight from the shard files through tensor handles, so a handle's
byte offset and row stride must describe exactly the FP8/E8M0 layout the
gather assumes. validate_engram_handles re-reads the shard headers to establish
that. Only metadata is read, the compiled extension is not imported and the
engine's shared safetensors loader is not changed.

read_header is stricter than a plain header parse: it bounds the header
length and rejects duplicate JSON keys, non-finite values, unknown dtype tags,
byte counts that disagree with the shape and overlapping tensor ranges.
"""

import json
import math
import os
import struct

MAX_HEADER_SIZE = 100 * 1024**2
_BITS = {"BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8,
         "U16": 16, "I16": 16, "F16": 16, "BF16": 16, "U32": 32, "I32": 32,
         "F32": 32, "U64": 64, "I64": 64, "F64": 64}


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate safetensors JSON key: {key}")
        value[key] = item
    return value


def _constant(value):
    raise ValueError(f"nonfinite safetensors JSON value: {value}")


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def read_header(path):
    """Return (header, data start, file size); never read tensor payloads.

    Unknown dtype tags are rejected, not guessed from byte width. The supported
    set is the fixed-width safetensors dtypes up to 64 bits, FP8 and E8M0
    included.
    """
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors header prefix: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if length < 2 or length > MAX_HEADER_SIZE or length > size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw, object_pairs_hook = _object, parse_constant = _constant)
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {path}")
    base, spans = 8 + length, []
    for key, entry in header.items():
        if key == "__metadata__":
            if not isinstance(entry, dict) or not all(isinstance(v, str) for v in entry.values()):
                raise ValueError(f"invalid safetensors metadata: {path}")
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"invalid tensor descriptor for {key}: {path}")
        shape, offsets, dtype = entry.get("shape"), entry.get("data_offsets"), entry.get("dtype")
        if not isinstance(shape, list) or not all(_integer(v) and v >= 0 for v in shape):
            raise ValueError(f"invalid tensor shape for {key}: {path}")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(_integer(v) for v in offsets):
            raise ValueError(f"invalid tensor offsets for {key}: {path}")
        lo, hi = offsets
        if not 0 <= lo <= hi <= size - base:
            raise ValueError(f"tensor range outside the file for {key}: {path}")
        if not isinstance(dtype, str) or dtype not in _BITS:
            raise ValueError(f"unsupported tensor dtype for {key}: {path}")
        expected = (math.prod(shape) * _BITS[dtype] + 7) // 8
        if hi - lo != expected:
            raise ValueError(f"tensor shape/byte count mismatch for {key}: {path}")
        if hi > lo:
            spans.append((lo, hi, key))
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            raise ValueError(f"overlapping tensor ranges for {left[2]} and {right[2]}: {path}")
    return header, base, size


def validate_engram_layout(weight, scale, head_dim, expected_rows = None):
    """Require the exact FP8/E8M0 32-value block layout, not just equal row bytes."""
    if not _integer(head_dim) or head_dim <= 0 or head_dim % 32:
        raise ValueError("engram head_dim must be a positive multiple of 32")
    wshape, sshape = weight.get("shape"), scale.get("shape")
    if weight.get("dtype") != "F8_E4M3" or scale.get("dtype") != "F8_E8M0":
        raise ValueError("engram table requires F8_E4M3 weights and F8_E8M0 scales")
    if not isinstance(wshape, list) or len(wshape) != 2 or not _integer(wshape[0]) or wshape[0] <= 0:
        raise ValueError("engram weight must have shape [positive rows, head_dim]")
    rows = wshape[0]
    if wshape != [rows, head_dim] or sshape != [rows, head_dim // 32]:
        raise ValueError("engram weight/scale shape does not match 32-value blocks")
    if expected_rows is not None and rows != expected_rows:
        raise ValueError("engram table row count does not match the configured bucket layout")
    return rows


def validate_engram_handles(weight, scale, head_dim, expected_rows):
    """Recheck disk handle metadata before the raw-byte gather path can use it."""
    headers, descriptors = {}, []
    for handle in (weight, scale):
        if handle.filename not in headers:
            headers[handle.filename] = read_header(handle.filename)
        header, base, _ = headers[handle.filename]
        descriptor = header.get(handle.key)
        if descriptor is None:
            raise ValueError("engram handle tensor is missing from its shard")
        if list(handle.shape) != descriptor["shape"] or handle.abs_offset != base + descriptor["data_offsets"][0]:
            raise ValueError("engram handle metadata disagrees with the shard")
        descriptors.append(descriptor)
    rows = validate_engram_layout(*descriptors, head_dim, expected_rows)
    if weight.row_bytes != head_dim or scale.row_bytes != head_dim // 32 \
            or weight.num_rows != rows or scale.num_rows != rows:
        raise ValueError("engram handle stride/row count disagrees with the table layout")
    return rows
