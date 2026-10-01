# shard_io.py
"""
Frame shard I/O shared by preprocess_dataset.py and every reader of `frames_dir/<task>/<task>_shard####.*`.

Two on-disk shard formats are supported:

  raw     `<task>_shard####.pt`      {"frames": (N, 3, H, W) uint8} via torch.save
  webp    `<task>_shard####.chunks`  frames grouped into fixed-size chunks; each chunk
                                     is the horizontal strip (H, n*W, 3) of `n` frames
                                     encoded as lossless WebP. Bit-identical to raw,
                                     roughly 10-30x smaller (content-dependent), and
                                     readers decode only the chunks overlapping a
                                     requested window instead of loading the whole shard.

`.chunks` file layout:
  MAGIC (8 bytes) | header_len uint32 LE | header JSON (utf-8) | chunk blobs (concatenated)
  header = {"format": "wm-chunked", "version": 1, "codec": "webp", "num_frames": N,
            "H": H, "W": W, "chunk_frames": C, "chunks": [[offset, nbytes, n_frames], ...],
            "crc32": [crc, ...]}
  offsets are relative to the first blob byte. "crc32" holds one CRC-32 per chunk blob and
  is optional: files written before it was added have no checksums and still load.

`load_shard()` returns a torch uint8 tensor for raw shards and a `ChunkedShard` for
chunked shards; both support `.shape`, `len()`, integer indexing, and contiguous slicing,
so callers can treat them interchangeably.
"""
import glob
import json
import os
import re
import struct
import zlib
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

MAGIC = b"WMCHUNK1"
RAW_EXT = ".pt"
CHUNKED_EXT = ".chunks"
DEFAULT_CHUNK_FRAMES = 16
CODECS = ("webp", "raw")

# libwebp refuses images wider than 16383 px, which caps chunk_frames at 73 for 224 px frames.
_WEBP_MAX_DIM = 16383


def shard_ext(codec: str) -> str:
    if codec == "raw":
        return RAW_EXT
    if codec == "webp":
        return CHUNKED_EXT
    raise ValueError(f"unknown codec {codec!r}; expected one of {CODECS}")


def check_codec(codec: str, chunk_frames: int = DEFAULT_CHUNK_FRAMES, frame_size: int = 224) -> None:
    """
    Raise if `write_shard` could not write full chunks of `frame_size` px frames with this
    configuration. Call once up front so a bad setting fails before any data is read.
    """
    shard_ext(codec)
    if codec != "webp":
        return
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be >= 1")
    if chunk_frames * frame_size > _WEBP_MAX_DIM:
        raise ValueError(
            f"chunk_frames={chunk_frames} at {frame_size} px gives a {chunk_frames * frame_size} px strip, "
            f"above the WebP limit of {_WEBP_MAX_DIM} px; use chunk_frames <= {_WEBP_MAX_DIM // frame_size}"
        )
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"codec 'webp' needs OpenCV, which failed to import ({e}); "
            "install opencv-python (see environment.yaml) or use codec 'raw'"
        ) from e


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
        "crc32": [zlib.crc32(b) for b in blobs],
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
    and returns a fresh uint8 tensor. Only the header is held between reads (the file is
    opened per read), so instances can be cached in any number and are safe to pickle or
    share with forked DataLoader workers.
    """

    ndim = 4
    dtype = torch.uint8

    def __init__(self, path: Union[str, Path]):
        self.path = str(path)
        with open(self.path, "rb") as f:
            magic = f.read(len(MAGIC))
            if magic != MAGIC:
                raise ValueError(f"{self.path}: not a chunked shard (bad magic {magic!r})")
            try:
                (hlen,) = struct.unpack("<I", f.read(4))
                header = json.loads(f.read(hlen).decode("utf-8"))
                kind = (header.get("format"), header.get("codec"), header.get("version"))
                self.num_frames = int(header["num_frames"])
                self.H, self.W = int(header["H"]), int(header["W"])
                self.chunk_frames = int(header["chunk_frames"])
                self._chunks: List[Tuple[int, int, int]] = [(int(o), int(nb), int(n)) for o, nb, n in header["chunks"]]
                crcs = header.get("crc32")
                self._crcs: Optional[List[int]] = None if crcs is None else [int(c) for c in crcs]
            except (struct.error, ValueError, KeyError, TypeError, AttributeError) as e:
                raise ValueError(f"{self.path}: corrupt header ({type(e).__name__}: {e})") from e
            file_size = os.fstat(f.fileno()).st_size
        if kind != ("wm-chunked", "webp", 1):
            raise ValueError(f"{self.path}: unsupported header (format, codec, version) = {kind}")
        self._blob_start = len(MAGIC) + 4 + hlen
        starts = np.zeros(len(self._chunks) + 1, dtype=np.int64)
        starts[1:] = np.cumsum([c[2] for c in self._chunks])
        if int(starts[-1]) != self.num_frames:
            raise ValueError(f"{self.path}: chunk frame counts do not sum to num_frames")
        if self._crcs is not None and len(self._crcs) != len(self._chunks):
            raise ValueError(f"{self.path}: corrupt header (crc32 does not match the chunk table)")
        expected_size = self._blob_start + max((o + nb for o, nb, _ in self._chunks), default=0)
        if file_size < expected_size:
            raise ValueError(f"{self.path}: truncated ({file_size} bytes on disk, header expects {expected_size})")
        self._starts = starts
        # Rough accounting for byte-budgeted caches (WMDataset.cache_mb): only the header
        # stays resident.
        self.nbytes = len(self._chunks) * 24 + 256

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.num_frames, 3, self.H, self.W)

    def __len__(self) -> int:
        return self.num_frames

    def _read_chunks(self, c0: int, c1: int) -> List[bytes]:
        # Open per call instead of keeping a descriptor: readers are cached per shard
        # (WMDataset never evicts them, see `nbytes`), so one held descriptor each would
        # exhaust RLIMIT_NOFILE on datasets with thousands of shards.
        fd = os.open(self.path, os.O_RDONLY)
        try:
            blobs = []
            for ci in range(c0, c1 + 1):
                off, nb, _ = self._chunks[ci]
                blob = os.pread(fd, nb, self._blob_start + off)
                if len(blob) != nb:
                    raise IOError(f"{self.path}: short read at offset {off} ({len(blob)}/{nb} bytes)")
                # Lossless WebP often still decodes after a bit flip, to wrong pixels.
                if self._crcs is not None and zlib.crc32(blob) != self._crcs[ci]:
                    raise IOError(f"{self.path}: checksum mismatch in chunk {ci} (file is corrupt)")
                blobs.append(blob)
            return blobs
        finally:
            os.close(fd)

    def read_range(self, start: int, end: int) -> torch.Tensor:
        """Decode frames [start, end) -> (end-start, 3, H, W) uint8 tensor."""
        if not (0 <= start <= end <= self.num_frames):
            raise IndexError(f"range [{start}, {end}) out of bounds for {self.num_frames} frames")
        if start == end:
            return torch.empty((0, 3, self.H, self.W), dtype=torch.uint8)
        c0 = int(np.searchsorted(self._starts, start, side="right") - 1)
        c1 = int(np.searchsorted(self._starts, end - 1, side="right") - 1)
        parts = []
        for ci, blob in zip(range(c0, c1 + 1), self._read_chunks(c0, c1)):
            n = self._chunks[ci][2]
            arr = _decode_chunk_webp(blob, n, self.H, self.W)
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
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
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


_SHARD_INDEX = re.compile(r"_shard(\d+)\.[a-z]+$")


def _check_contiguous(paths: List[str], task_dir: str) -> None:
    # Readers concatenate shards positionally (WMDataset, plan_cem), so a gap left by a failed
    # or deleted shard would shift every later frame against its action and reward.
    matches = [_SHARD_INDEX.search(os.path.basename(p)) for p in paths]
    if not all(matches):
        return  # not named by preprocess_dataset.py; nothing to check against
    present = {int(m.group(1)) for m in matches}
    missing = sorted(set(range(max(present, default=-1) + 1)) - present)
    if missing:
        raise ValueError(
            f"{task_dir}: shard indices are not contiguous (missing {missing[:5]}), which would misalign "
            "frames with actions/rewards; rerun preprocess_dataset.py for this task"
        )


def list_shards(task_dir: Union[str, Path]) -> List[str]:
    """
    Sorted shard paths in a task directory (either format).

    If `<task>_index.json` exists it is authoritative: preprocess_dataset.py writes it last,
    so it names exactly the shards of the completed run, and leftovers of an interrupted run
    in the other format are ignored. Without an index, a directory holding both formats is
    ambiguous (at least one of them is partial) and is refused. A gap in the shard numbering
    is refused either way; missing trailing shards only truncate the task.
    """
    task_dir = str(task_dir)
    task = os.path.basename(os.path.normpath(task_dir))
    index_path = os.path.join(task_dir, f"{task}_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                names = sorted(json.load(f))
        except (OSError, ValueError, TypeError) as e:
            print(f"[shard_io] ignoring unreadable index {index_path}: {e}")
        else:
            paths = [os.path.join(task_dir, name) for name in names]
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                print(f"[shard_io] {len(missing)} shards listed in {index_path} are missing (e.g. {missing[0]})")
            paths = [p for p in paths if os.path.exists(p)]
            _check_contiguous(paths, task_dir)
            return paths
    chunked = sorted(glob.glob(os.path.join(task_dir, "*_shard*" + CHUNKED_EXT)))
    raw = sorted(glob.glob(os.path.join(task_dir, "*_shard*" + RAW_EXT)))
    if chunked and raw:
        raise ValueError(
            f"{task_dir} holds both {CHUNKED_EXT} and {RAW_EXT} shards and no {task}_index.json, so at "
            "least one set is partial; rerun preprocess_dataset.py for this task or delete one format"
        )
    _check_contiguous(chunked or raw, task_dir)
    return chunked or raw
