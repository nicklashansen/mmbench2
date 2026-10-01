# test_shard_io.py
"""
Tests for the frame shard formats (src/shard_io.py) and their use in preprocessing and
the dataset classes. Synthetic data only; CPU only; takes well under a minute.

Run from the repo root:
    python -m pytest tests -q
"""
import io
import json
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, SRC)

import preprocess_dataset  # noqa: E402
import shard_io  # noqa: E402
from shard_io import ChunkedShard, check_codec, list_shards, load_shard, shard_filename, write_shard  # noqa: E402
from sharded_frame_dataset import ShardedFrameDataset  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402

TASK = "toy-task"
T = 8                    # WMDataset seq_len
EP_LEN = 20              # frames per synthetic episode
STRIPS = (40, 40, 23)    # frames per source PNG strip
SHARD_SIZE = 32          # -> shards of 32, 32, 32, 7 frames
ACT_DIM = 4


def make_frames(n, h=224, w=224, seed=0):
    """(n, 3, h, w) uint8 blocky frames: every frame and every channel is distinct."""
    g = torch.Generator().manual_seed(seed)
    low = torch.randint(0, 256, (n, 3, max(1, h // 16), max(1, w // 16)), generator=g, dtype=torch.uint8)
    frames = torch.nn.functional.interpolate(low.float(), size=(h, w), mode="nearest").to(torch.uint8)
    frames[:, 0, 0, 0] = torch.arange(n) % 256
    return frames


def num_open_fds():
    return len(os.listdir("/proc/self/fd"))


needs_proc_fd = pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc/self/fd")


def run_preprocess(filedir, outdir, codec, chunk_frames=16, shard_size=SHARD_SIZE):
    return preprocess_dataset.process_task((TASK, str(filedir), str(outdir), 224, shard_size, codec, chunk_frames))


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """A one-task dataset in the on-disk layout of the real one, preprocessed in both formats."""
    from torchvision.io import write_png

    root = tmp_path_factory.mktemp("toy")
    raw_dir = root / "data"
    raw_dir.mkdir()
    frames = make_frames(sum(STRIPS))
    s = 0
    for i, n in enumerate(STRIPS):
        strip = frames[s:s + n].permute(1, 2, 0, 3).reshape(3, 224, n * 224)   # frames side by side
        write_png(strip, str(raw_dir / f"{TASK}-{i}.png"))
        s += n
    N = frames.shape[0]
    g = torch.Generator().manual_seed(1)
    episode = torch.arange(N) // EP_LEN
    action = torch.rand(N, ACT_DIM, generator=g)
    reward = torch.rand(N, generator=g)
    first = torch.arange(N) % EP_LEN == 0           # (obs_0, nan, nan) convention
    action[first] = float("nan")
    reward[first] = float("nan")
    torch.save({"episode": episode, "action": action, "reward": reward}, raw_dir / f"{TASK}.pt")
    for codec in ("webp", "raw"):
        assert run_preprocess(raw_dir, root / f"shards-{codec}", codec) == 0
    return Namespace(root=root, data=str(raw_dir), webp=str(root / "shards-webp"), raw=str(root / "shards-raw"),
                     frames=frames, action=action, reward=reward)


def wm_dataset(data_dir, frames_dir, **kw):
    return WMDataset(data_dir, frames_dir, seq_len=T, img_size=224, action_dim=ACT_DIM, tasks_json=None,
                     tasks=[TASK], verbose=False, **kw)


# ----------------------------------------------------------------------------- format

@pytest.mark.parametrize("n,chunk_frames,h,w", [
    (0, 16, 32, 32), (1, 16, 32, 32), (15, 16, 32, 32), (16, 16, 32, 32), (17, 16, 32, 32),
    (50, 1, 32, 32), (50, 7, 32, 32), (50, 64, 32, 32), (20, 16, 24, 40), (5, 16, 1, 1), (33, 16, 224, 224),
])
def test_roundtrip_is_bit_exact(tmp_path, n, chunk_frames, h, w):
    frames = make_frames(n, h, w)
    assert write_shard(frames, tmp_path / shard_filename("t", 0, "webp"), chunk_frames=chunk_frames) == n
    write_shard(frames, tmp_path / shard_filename("t", 0, "raw"), codec="raw")
    chunked, raw = load_shard(tmp_path / "t_shard0000.chunks"), load_shard(tmp_path / "t_shard0000.pt")
    assert isinstance(chunked, ChunkedShard) and isinstance(raw, torch.Tensor)
    assert tuple(chunked.shape) == tuple(raw.shape) == (n, 3, h, w) and len(chunked) == n
    assert chunked.ndim == 4 and chunked.dtype == torch.uint8
    assert torch.equal(chunked[:], frames) and torch.equal(raw, frames)
    rng = np.random.default_rng(0)
    for _ in range(20 if n else 0):
        a, b = sorted(int(x) for x in rng.integers(0, n + 1, size=2))
        assert torch.equal(chunked[a:b], frames[a:b])
        i = int(rng.integers(0, n))
        assert torch.equal(chunked[i], frames[i]) and torch.equal(chunked[np.int64(i)], frames[i])


def test_indexing_matches_tensor(tmp_path):
    n = 50
    frames = make_frames(n, 16, 16)
    write_shard(frames, tmp_path / "t_shard0000.chunks")
    shard = load_shard(tmp_path / "t_shard0000.chunks")
    for i in range(-n - 2, n + 2):
        if -n <= i < n:
            assert torch.equal(shard[i], frames[i])
        else:
            with pytest.raises(IndexError):
                shard[i]
    bounds = [None, -n - 3, -n, -17, -1, 0, 1, 15, 16, 17, 32, n - 1, n, n + 3]
    for a in bounds:
        for b in bounds:
            assert torch.equal(shard[a:b], frames[a:b]), (a, b)
    for bad in (slice(None, None, 2), slice(None, None, -1)):
        with pytest.raises(IndexError):
            shard[bad]
    with pytest.raises(TypeError):
        shard[[0, 1]]
    out = shard[3:9]
    assert out.is_contiguous() and out.dtype == torch.uint8
    out.zero_()                                           # results are independent copies
    assert torch.equal(shard[3:9], frames[3:9])


def test_on_disk_layout_is_rgb_strip(tmp_path):
    """Pin the documented layout with an independent decoder: each chunk is the RGB strip (H, n*W, 3)."""
    Image = pytest.importorskip("PIL.Image")
    n, h, w, cf = 10, 24, 40, 4
    frames = make_frames(n, h, w)
    path = tmp_path / "t_shard0000.chunks"
    write_shard(frames, path, chunk_frames=cf)
    data = path.read_bytes()
    assert data[:8] == b"WMCHUNK1"
    hlen = int.from_bytes(data[8:12], "little")
    header = json.loads(data[12:12 + hlen])
    assert (header["num_frames"], header["H"], header["W"], header["chunk_frames"]) == (n, h, w, cf)
    assert [c[2] for c in header["chunks"]] == [4, 4, 2]
    start = 0
    for off, nbytes, k in header["chunks"]:
        blob = data[12 + hlen + off:12 + hlen + off + nbytes]
        strip = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
        assert strip.shape == (h, k * w, 3)
        for j in range(k):
            assert np.array_equal(strip[:, j * w:(j + 1) * w], frames[start + j].permute(1, 2, 0).numpy())
        start += k


def test_write_shard_validates_and_is_atomic(tmp_path, monkeypatch):
    frames = make_frames(40, 16, 16)
    with pytest.raises(ValueError):
        write_shard(frames.float(), tmp_path / "t_shard0000.chunks")
    with pytest.raises(ValueError):
        write_shard(frames.permute(0, 2, 3, 1), tmp_path / "t_shard0000.chunks")
    with pytest.raises(ValueError):
        write_shard(frames, tmp_path / "t_shard0000.pt", codec="webp")     # extension/codec mismatch
    with pytest.raises(ValueError):
        write_shard(make_frames(80, 224, 224), tmp_path / "t_shard0000.chunks", chunk_frames=74)
    assert list(tmp_path.iterdir()) == []

    path = tmp_path / "t_shard0000.chunks"
    write_shard(frames, path)
    real, calls = shard_io._encode_chunk_webp, []

    def flaky(x):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(x)

    monkeypatch.setattr(shard_io, "_encode_chunk_webp", flaky)
    with pytest.raises(RuntimeError):
        write_shard(make_frames(40, 16, 16, seed=5), path)
    monkeypatch.undo()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t_shard0000.chunks"]   # no .tmp left behind
    assert torch.equal(load_shard(path)[:], frames)                              # old shard intact


def test_raw_codec_saves_only_the_slice(tmp_path):
    big = make_frames(64, 32, 32)
    view = big[8:16]
    for codec in ("raw", "webp"):
        p = tmp_path / shard_filename("t", 0, codec)
        write_shard(view, p, codec=codec)
        assert torch.equal(load_shard(p)[:], view)
    # torch.save would serialize the whole backing storage of a view without the clone()
    assert os.path.getsize(tmp_path / "t_shard0000.pt") < 2 * view.numel()


def test_corrupt_files_raise(tmp_path):
    path = tmp_path / "t_shard0000.chunks"
    write_shard(make_frames(40, 16, 16), path)
    data = path.read_bytes()
    bad = tmp_path / "bad_shard0000.chunks"
    bad.write_bytes(b"NOTMAGIC" + data[8:])
    with pytest.raises(ValueError):
        ChunkedShard(bad)
    bad.write_bytes(data[:-1])                            # truncated inside the last chunk
    shard = ChunkedShard(bad)
    assert torch.equal(shard[0:16], ChunkedShard(path)[0:16])
    with pytest.raises(IOError):
        shard[:]
    with pytest.raises(ValueError):
        load_shard(tmp_path / "t_shard0000.bin")


# ----------------------------------------------------------------------------- reader lifetime

@needs_proc_fd
def test_readers_hold_no_file_descriptors(tmp_path):
    frames = make_frames(8, 16, 16)
    for i in range(64):
        write_shard(frames, tmp_path / shard_filename("t", i, "webp"))
    before = num_open_fds()
    shards = [load_shard(p) for p in list_shards(tmp_path)]
    assert len(shards) == 64
    for s in shards:                                      # all readers stay referenced
        assert torch.equal(s[2:6], frames[2:6])
    assert num_open_fds() == before
    clone = pickle.loads(pickle.dumps(shards[0]))         # after a read
    assert torch.equal(clone[:], frames) and num_open_fds() == before


@needs_proc_fd
def test_wm_dataset_cache_does_not_leak_descriptors(toy):
    ds = wm_dataset(toy.data, toy.webp)
    before = num_open_fds()
    for i in range(len(ds)):
        ds[i]
    assert len(ds._cache) == 4                            # one cached reader per shard, never evicted
    assert num_open_fds() == before


# ----------------------------------------------------------------------------- preprocessing

def test_preprocess_formats_match_source(toy):
    for d, ext in ((toy.webp, ".chunks"), (toy.raw, ".pt")):
        index = json.load(open(os.path.join(d, TASK, f"{TASK}_index.json")))
        assert list(index) == [f"{TASK}_shard{i:04d}{ext}" for i in range(4)]
        assert list(index.values()) == [32, 32, 32, 7]
        paths = list_shards(os.path.join(d, TASK))
        assert [os.path.basename(p) for p in paths] == list(index)
        assert torch.equal(torch.cat([load_shard(p)[:] for p in paths]), toy.frames)


def test_check_codec():
    check_codec("raw", 0, 224)
    check_codec("webp", 73, 224)
    check_codec("webp", 127, 128)
    for bad in (("webp", 74, 224), ("webp", 128, 128), ("webp", 0, 224), ("png", 16, 224)):
        with pytest.raises(ValueError):
            check_codec(*bad)


def test_preprocess_rejects_bad_config_before_reading(toy, tmp_path):
    out = tmp_path / "out"
    args = Namespace(filedir=toy.data, outdir=str(out), target_size=224, shard_size=SHARD_SIZE, codec="webp",
                     chunk_frames=100, num_workers=1, tasks=[TASK], task_set="trained")
    with pytest.raises(SystemExit) as e:
        preprocess_dataset.main(args)
    assert "chunk_frames" in str(e.value) and not out.exists()


def test_failed_save_withholds_index_and_rerun_completes(toy, tmp_path, monkeypatch):
    out = tmp_path / "out"
    real, calls = preprocess_dataset.write_shard, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(preprocess_dataset, "write_shard", flaky)
    assert run_preprocess(toy.data, out, "webp") == 1
    assert not (out / TASK / f"{TASK}_index.json").exists()
    monkeypatch.undo()
    assert run_preprocess(toy.data, out, "webp") == 0
    assert torch.equal(torch.cat([load_shard(p)[:] for p in list_shards(out / TASK)]), toy.frames)


# ----------------------------------------------------------------------------- datasets

def test_sharded_frame_dataset_matches_across_formats(toy):
    a = ShardedFrameDataset(toy.webp, tasks=[TASK], seq_len=T, iid_sampling=False, verbose=False)
    b = ShardedFrameDataset(toy.raw, tasks=[TASK], seq_len=T, iid_sampling=False, verbose=False)
    assert len(a) == len(b) == sum(n - T + 1 for n in (32, 32, 32)) and a.cum_starts == b.cum_starts
    for i in range(len(a)):
        shard_idx, start = a._map_global_start_to_shard(i)
        want = toy.frames[shard_idx * SHARD_SIZE + start:shard_idx * SHARD_SIZE + start + T].float() / 255.0
        assert torch.equal(a[i], want) and torch.equal(b[i], want)


def test_wm_dataset_matches_source_and_convention(toy):
    a, b = wm_dataset(toy.data, toy.webp), wm_dataset(toy.data, toy.raw)
    assert len(a) == len(b) > 0
    crossing = 0
    for i in range(len(a)):
        _, start = a._lookup(i)
        x, y = a[i], b[i]
        # obs_t pairs with (action_{t-1}, reward_{t-1}): window frames start..start+T use transitions start+1..start+T
        assert torch.equal(x["obs"], toy.frames[start:start + T + 1]) and torch.equal(y["obs"], x["obs"])
        assert torch.equal(x["act"], toy.action[start + 1:start + 1 + T]) and torch.equal(y["act"], x["act"])
        assert torch.equal(x["rew"], toy.reward[start + 1:start + 1 + T]) and torch.equal(y["rew"], x["rew"])
        crossing += start // SHARD_SIZE != (start + T) // SHARD_SIZE
    assert crossing > 0                                   # some windows straddle a shard boundary


@pytest.mark.parametrize("ctx", ["fork", "spawn"])
def test_multi_worker_loader_matches_in_process(toy, ctx, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))   # for spawned workers
    ds, ref = wm_dataset(toy.data, toy.webp), wm_dataset(toy.data, toy.raw)
    ds[0]                                                 # a reader is already cached when workers start
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_batch,
                        multiprocessing_context=ctx)
    seen = 0
    for bi, batch in enumerate(loader):
        want = collate_batch([ref[i] for i in range(bi * 4, min(len(ref), bi * 4 + 4))])
        assert all(torch.equal(batch[k], want[k]) for k in want)
        seen += len(batch["obs"])
    assert seen == len(ds)
    frames = ShardedFrameDataset(toy.webp, tasks=[TASK], seq_len=T, iid_sampling=True, samples_per_shard=3, verbose=False)
    loader = DataLoader(frames, batch_size=4, num_workers=2, multiprocessing_context=ctx)
    assert next(iter(loader)).shape == (4, T, 3, 224, 224)


# ----------------------------------------------------------------------------- mixed-format directories

def test_index_is_authoritative_in_mixed_directory(toy, tmp_path):
    """Leftovers of an interrupted run in the other format must not shadow the completed run."""
    import shutil

    mixed = tmp_path / "shards"
    shutil.copytree(toy.raw, mixed)
    shutil.copy(os.path.join(toy.webp, TASK, f"{TASK}_shard0000.chunks"), mixed / TASK)   # stale partial webp run
    assert [os.path.basename(p) for p in list_shards(mixed / TASK)] == [f"{TASK}_shard{i:04d}.pt" for i in range(4)]
    ds, ref = wm_dataset(toy.data, str(mixed)), wm_dataset(toy.data, toy.raw)
    assert len(ds) == len(ref) and ds.seg_cum_frames == ref.seg_cum_frames == [[sum(STRIPS)]]
    assert all(p.endswith(".pt") for p in ds.shard_lists[0][0])
    sf = ShardedFrameDataset(str(mixed), tasks=[TASK], seq_len=T, iid_sampling=False, verbose=False)
    assert [s["num_frames"] for s in sf.shards] == [32, 32, 32]


def test_mixed_directory_without_index_is_refused(toy, tmp_path):
    import shutil

    mixed = tmp_path / "shards"
    shutil.copytree(toy.raw, mixed)
    shutil.copy(os.path.join(toy.webp, TASK, f"{TASK}_shard0000.chunks"), mixed / TASK)
    os.remove(mixed / TASK / f"{TASK}_index.json")
    with pytest.raises(ValueError, match="both"):
        list_shards(mixed / TASK)
    with pytest.raises(ValueError, match="both"):
        wm_dataset(toy.data, str(mixed))
    with pytest.raises(ValueError, match="both"):
        ShardedFrameDataset(str(mixed), tasks=[TASK], seq_len=T, iid_sampling=False, verbose=False)
    # a single format without an index still works (slow scan), and an unreadable index is ignored
    os.remove(mixed / TASK / f"{TASK}_shard0000.chunks")
    (mixed / TASK / f"{TASK}_index.json").write_text("{not json")
    assert len(list_shards(mixed / TASK)) == 4
    assert len(wm_dataset(toy.data, str(mixed))) == len(wm_dataset(toy.data, toy.raw))
