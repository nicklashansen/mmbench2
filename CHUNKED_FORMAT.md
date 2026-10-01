# Chunked frame shards

This branch adds a compressed on-disk format for preprocessed frame shards, replacing the
raw uint8 `torch.save` shards with **chunked lossless WebP**. Frames are bit-identical to
the raw format and the dataset is **roughly 10-30x smaller on disk**, depending on the
task (about 5-15 KB/frame instead of 150 KB at 224x224; e.g. 16-21x for `ms-pick-cube`
and 11x for `walker-walk`). Readers decode only the chunks a sampled window needs instead
of `torch.load`-ing whole ~600 MB shards. See [Performance](#performance) for what that
costs and when it pays off.

## Format

A shard file `<task>_shard####.chunks` stores up to `--shard_size` (default 4096) frames:

```
MAGIC "WMCHUNK1" | header_len uint32 LE | header JSON | chunk blobs (concatenated)
```

Frames are grouped into fixed-size chunks (default 16). Each chunk is the horizontal
strip `(H, n*W, 3)` of its `n` frames, encoded as lossless WebP. The JSON header lists
`[offset, nbytes, n_frames]` per chunk, so a reader `pread()`s and decodes only the
chunks overlapping a requested frame range. The `<task>_index.json` convention
(`{shard_name: num_frames}`, index written last = task complete) is unchanged, and the
legacy raw `.pt` format remains fully supported for both reading and writing.

## Usage

- **Preprocess** as before (`bash preprocess.sh` or `src/preprocess_dataset.py`); the
  chunked format is now the default. Pass `--codec raw` for legacy uint8 shards, and
  `--chunk_frames` to tune the chunk size (`chunk_frames * target_size` must stay within
  WebP's 16383 px limit, i.e. at most 73 frames at 224 px; this is checked up front).
- **Load** via `src/shard_io.py`: `load_shard(path)` returns a uint8 tensor (raw) or a
  lazy `ChunkedShard` (chunked) — both support `.shape`, `len()`, integer indexing, and
  unit-step slicing (`s[:]` materializes). `list_shards(task_dir)` lists either format.
  `ShardedFrameDataset` and `WMDataset` accept either format transparently, so existing
  training commands work unchanged.
- **Write** programmatically with `write_shard(frames_u8_NCHW, path, codec, chunk_frames)`
  (atomic; filenames via `shard_filename(task, idx, codec)`).

## Existing and mixed directories

- A task whose `<task>_index.json` already exists is skipped, so re-running preprocessing
  on an existing raw output directory does **not** convert it. To convert, preprocess into
  a fresh `--outdir` (or delete the old task directories first).
- When an index exists it is authoritative: readers use exactly the shards it lists and
  ignore leftovers of an interrupted run in the other format.
- Without an index, a task directory holding both `.chunks` and `.pt` shards is ambiguous
  and readers refuse it; re-run preprocessing for that task or delete one format.
- `preprocess_dataset.py` does not write a task's index if any shard save failed and exits
  non-zero, so re-running it resumes the missing tasks. A task directory without an index
  is incomplete: readers still accept it (slow scan), so finish preprocessing before
  training on it.

## Performance

Measured on `ms-pick-cube` and `walker-walk` at 224x224 with the default chunk size, on a
busy 64-core machine; treat the numbers as indicative.

- **Loading.** A 24-frame window costs about 20-35 ms of CPU to decode, because it
  overlaps 2-3 chunks of 16 frames. A raw reader instead pays ~0.4 s (plus disk time) to
  load a whole shard and nothing per window after that. Chunked shards therefore win when
  loading is I/O-bound (cold page cache, network storage, a dataset far larger than RAM)
  and lose when raw shards are served from a warm page cache: at the default loader
  settings we measured about 100-110 samples/s for chunked versus 160-230 samples/s for
  raw with a reload on every shard switch. If the data loader is the bottleneck with
  chunked shards, raise `--num_workers`.
- **Sampling.** Reading a window costs the same whichever shard it comes from, so
  `samples_per_shard=1` (fully i.i.d. batches) is no slower than the default — note it
  changes batch composition relative to the legacy shard-locality sampling.
- **Preprocessing.** Lossless WebP encoding is CPU-bound at about 20 ms per frame per
  core, so preprocessing takes roughly 15x longer than `--codec raw` (on the order of 100
  CPU-hours for the training partitions). Tasks are processed in parallel
  (`--num_workers`).

## Notes

- Both formats hold identical frames, so switching formats does not change training data.
  At the default `--target_size 224` they are also bit-identical to the source PNGs; other
  sizes are resized before saving.
- Decoding uses OpenCV (added to `environment.yaml`). WebP encoding is lossless
  (`quality=101` selects lossless mode in OpenCV's encoder).
- A `ChunkedShard` keeps only its header in memory and opens the file per read, so the
  dataset classes can cache any number of them; `--cache_mb` only bounds raw shards.

## Tests

- `python -m pytest tests -q` — synthetic round-trip, indexing, preprocessing, dataset,
  and multi-worker DataLoader tests for both formats (CPU, under a minute).
- `python tests/check_real_data.py --data_dir src/data --partitions val --tasks ms-pick-cube`
  — preprocesses real tasks in both formats and checks them against the source PNGs and
  against each other through both dataset classes.
- `python tests/bench_loading.py --frame_dirs src/data/val-shards --data_dirs src/data/val --tasks ms-pick-cube`
  — loader throughput at the training defaults for whichever format the directory holds.
