# shard_io.py
"""
Frame shard I/O shared by preprocess_dataset.py and every reader of `frames_dir/<task>/<task>_shard####.*`.

Two on-disk shard formats are supported:

  raw     `<task>_shard####.pt`      {"frames": (N, 3, H, W) uint8} via torch.save
  webp    `<task>_shard####.chunks`  frames grouped into fixed-size chunks; each chunk
                                     is the horizontal strip (H, n*W, 3) of `n` frames
                                     encoded as lossless WebP. Bit-identical to raw,
                                     ~20-30x smaller, and readers decode only the chunks
                                     overlapping a requested window instead of loading
                                     the whole shard.

`.chunks` file layout:
  MAGIC (8 bytes) | header_len uint32 LE | header JSON (utf-8) | chunk blobs (concatenated)
  header = {"format": "wm-chunked", "version": 1, "codec": "webp", "num_frames": N,
            "H": H, "W": W, "chunk_frames": C, "chunks": [[offset, nbytes, n_frames], ...]}
  offsets are relative to the first blob byte.

`load_shard()` returns a torch uint8 tensor for raw shards and a `ChunkedShard` for
chunked shards; both support `.shape`, `len()`, integer indexing, and contiguous slicing,
so callers can treat them interchangeably.
"""
import glob
import json
import os
import struct
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

MAGIC = b"WMCHUNK1"
RAW_EXT = ".pt"
CHUNKED_EXT = ".chunks"
DEFAULT_CHUNK_FRAMES = 16
CODECS = ("webp", "raw")

# libwebp refuses images wider than 16383 px, which caps chunk_frames at 224 px.
_WEBP_MAX_DIM = 16383


def shard_ext(codec: str) -> str:
    if codec == "raw":
        return RAW_EXT
    if codec == "webp":
        return CHUNKED_EXT
    raise ValueError(f"unknown codec {codec!r}; expected one of {CODECS}")


def shard_filename(task: str, shard_idx: int, codec: str) -> str:
    return f"{task}_shard{shard_idx:04d}{shard_ext(codec)}"


# --------------------------------------------------------------------------- encode

def _encode_chunk_webp(frames: torch.Tensor) -> bytes:
    """(n, 3, H, W) uint8 -> lossless WebP bytes of the (H, n*W, 3) strip."""
    import cv2

    n, _, H, W = frames.shape
    if n * W > _WEBP_MAX_DIM:
        raise ValueError(f"chunk strip width {n * W} exceeds WebP limit {_WEBP_MAX_DIM}")
    strip = frames.permute(2, 0, 3, 1).reshape(H, n * W, 3).numpy()  # RGB
    bgr = np.ascontiguousarray(strip[:, :, ::-1])
    # quality > 100 selects lossless mode in OpenCV's WebP encoder.
    ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, 101])
    if not ok:
        raise RuntimeError("cv2.imencode(.webp) failed")
    return buf.tobytes()


def _decode_chunk_webp(blob: bytes, n: int, H: int, W: int) -> np.ndarray:
    """WebP bytes -> (n, 3, H, W) uint8 RGB numpy array (not necessarily contiguous)."""
    import cv2

    bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (H, n * W, 3):
        got = None if bgr is None else bgr.shape
        raise RuntimeError(f"chunk decode failed: expected {(H, n * W, 3)}, got {got}")
    return bgr.reshape(H, n, W, 3).transpose(1, 3, 0, 2)[:, ::-1]


def _as_uint8_nchw(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected (N, 3, H, W) frames, got {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        raise ValueError(f"expected uint8 frames, got {frames.dtype}")
    return frames.contiguous()


def write_shard(
    frames: torch.Tensor,
    out_path: Union[str, Path],
    codec: str = "webp",
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> int:
    """
    Write (N, 3, H, W) uint8 frames to `out_path` in the given codec. Writes to a temp
    file and renames atomically. Returns the number of frames written.

    `out_path` must carry the codec's extension (see `shard_filename`).
    """
    out_path = Path(out_path)
    if out_path.suffix != shard_ext(codec):
        raise ValueError(f"{out_path.name} does not have the {codec!r} extension {shard_ext(codec)}")
    frames = _as_uint8_nchw(frames)
    N = int(frames.shape[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        if codec == "raw":
            # clone(): torch.save serializes the whole backing storage, so saving a
            # slice of a larger tensor would write the unrelated remainder too.
            torch.save({"frames": frames.clone()}, tmp_path)
        else:
            _write_chunked(frames, tmp_path, codec, int(chunk_frames))
        os.replace(tmp_path, out_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return N


def _write_chunked(frames: torch.Tensor, path: Path, codec: str, chunk_frames: int) -> None:
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be >= 1")
    N, _, H, W = frames.shape
    chunks: List[List[int]] = []
    blobs: List[bytes] = []
    off = 0
    for s in range(0, N, chunk_frames):
        n = min(chunk_frames, N - s)
        blob = _encode_chunk_webp(frames[s:s + n])
        chunks.append([off, len(blob), n])
        blobs.append(blob)
        off += len(blob)
    header = {
        "format": "wm-chunked", "version": 1, "codec": codec,
        "num_frames": int(N), "H": int(H), "W": int(W),
        "chunk_frames": int(chunk_frames), "chunks": chunks,
    }
    hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hbytes)))
        f.write(hbytes)
        for b in blobs:
            f.write(b)


# --------------------------------------------------------------------------- decode

class ChunkedShard:
    """
    Lazy reader for a `.chunks` shard. Mimics the subset of the tensor interface that
    the dataset classes use: `.shape`, `.ndim`, `.dtype`, `len()`, `shard[i]`, and
    `shard[a:b]` (unit step). Indexing decodes only the chunks that overlap the request
    and returns a fresh uint8 tensor. The file descriptor is opened lazily per process,
    so instances are safe to create before forking DataLoader workers.
    """

    ndim = 4
    dtype = torch.uint8

    def __init__(self, path: Union[str, Path]):
        self.path = str(path)
        with open(self.path, "rb") as f:
            magic = f.read(len(MAGIC))
            if magic != MAGIC:
                raise ValueError(f"{self.path}: not a chunked shard (bad magic {magic!r})")
            (hlen,) = struct.unpack("<I", f.read(4))
            header = json.loads(f.read(hlen).decode("utf-8"))
        if header.get("format") != "wm-chunked" or header.get("codec") != "webp":
            raise ValueError(f"{self.path}: unsupported header {header.get('format')}/{header.get('codec')}")
        self._blob_start = len(MAGIC) + 4 + hlen
        self.num_frames = int(header["num_frames"])
        self.H, self.W = int(header["H"]), int(header["W"])
        self.chunk_frames = int(header["chunk_frames"])
        self._chunks: List[Tuple[int, int, int]] = [tuple(c) for c in header["chunks"]]
        starts = np.zeros(len(self._chunks) + 1, dtype=np.int64)
        starts[1:] = np.cumsum([c[2] for c in self._chunks])
        if int(starts[-1]) != self.num_frames:
            raise ValueError(f"{self.path}: chunk frame counts do not sum to num_frames")
        self._starts = starts
        self._fd = None
        # Rough accounting for byte-budgeted caches (WMDataset.cache_mb): only the header
        # stays resident.
        self.nbytes = len(self._chunks) * 24 + 256

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.num_frames, 3, self.H, self.W)

    def __len__(self) -> int:
        return self.num_frames

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_fd"] = None
        return d

    def __del__(self):
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _read(self, offset: int, nbytes: int) -> bytes:
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
        out = os.pread(self._fd, nbytes, self._blob_start + offset)
        if len(out) != nbytes:
            raise IOError(f"{self.path}: short read at offset {offset} ({len(out)}/{nbytes} bytes)")
        return out

    def read_range(self, start: int, end: int) -> torch.Tensor:
        """Decode frames [start, end) -> (end-start, 3, H, W) uint8 tensor."""
        if not (0 <= start <= end <= self.num_frames):
            raise IndexError(f"range [{start}, {end}) out of bounds for {self.num_frames} frames")
        if start == end:
            return torch.empty((0, 3, self.H, self.W), dtype=torch.uint8)
        c0 = int(np.searchsorted(self._starts, start, side="right") - 1)
        c1 = int(np.searchsorted(self._starts, end - 1, side="right") - 1)
        parts = []
        for ci in range(c0, c1 + 1):
            off, nb, n = self._chunks[ci]
            arr = _decode_chunk_webp(self._read(off, nb), n, self.H, self.W)
            cs = int(self._starts[ci])
            lo, hi = max(start - cs, 0), min(end - cs, n)
            parts.append(arr[lo:hi])
        out = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        return torch.from_numpy(np.ascontiguousarray(out))

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.num_frames)
            if step != 1:
                raise IndexError("ChunkedShard only supports unit-step slices")
            return self.read_range(start, max(start, stop))
        if isinstance(key, (int, np.integer)):
            i = int(key)
            if i < 0:
                i += self.num_frames
            if not (0 <= i < self.num_frames):
                raise IndexError(i)
            return self.read_range(i, i + 1)[0]
        raise TypeError(f"unsupported index type {type(key).__name__}")


def load_shard(path: Union[str, Path]) -> Union[torch.Tensor, ChunkedShard]:
    """Open a shard of either format. Raw shards are loaded fully; chunked shards lazily."""
    path = str(path)
    if path.endswith(CHUNKED_EXT):
        return ChunkedShard(path)
    if path.endswith(RAW_EXT):
        td = torch.load(path, map_location="cpu", weights_only=True)
        return td["frames"]
    raise ValueError(f"unrecognized shard file {path}")


def list_shards(task_dir: Union[str, Path]) -> List[str]:
    """Sorted shard paths in a task directory (either format; chunked wins if both exist)."""
    task_dir = str(task_dir)
    chunked = sorted(glob.glob(os.path.join(task_dir, "*_shard*" + CHUNKED_EXT)))
    raw = sorted(glob.glob(os.path.join(task_dir, "*_shard*" + RAW_EXT)))
    if chunked and raw:
        print(f"[shard_io] {task_dir} holds both .chunks and .pt shards; using .chunks")
    return chunked or raw


def is_shard_path(name: str) -> bool:
    return "_shard" in os.path.basename(name) and (name.endswith(RAW_EXT) or name.endswith(CHUNKED_EXT))
