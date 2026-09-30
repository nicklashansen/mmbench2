# Chunked frame shards

This branch adds a compressed on-disk format for preprocessed frame shards, replacing the
raw uint8 `torch.save` shards with **chunked lossless WebP**. Frames are bit-identical to
the raw format, the dataset is **~20-30x smaller on disk** (roughly 3-10 KB/frame instead
of 150 KB at 224x224), and training data loading gets *faster*, because readers decode
only the frames a sampled window needs instead of `torch.load`-ing whole ~600 MB shards.

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
  `--chunk_frames` to tune the chunk size.
- **Load** via `src/shard_io.py`: `load_shard(path)` returns a uint8 tensor (raw) or a
  lazy `ChunkedShard` (chunked) — both support `.shape`, `len()`, integer indexing, and
  unit-step slicing (`s[:]` materializes). `list_shards(task_dir)` lists either format.
  `ShardedFrameDataset` and `WMDataset` accept either format transparently, so existing
  training commands work unchanged.
- **Write** programmatically with `write_shard(frames_u8_NCHW, path, codec, chunk_frames)`
  (atomic; filenames via `shard_filename(task, idx, codec)`).

## Notes

- Both formats are bit-identical to the source PNGs; switching formats does not change
  training data. With cheap per-window decoding, `samples_per_shard=1` (fully i.i.d.
  batches) is affordable — note it changes batch composition relative to the legacy
  shard-locality sampling.
- Decoding uses OpenCV (added to `environment.yaml`). WebP encoding is lossless
  (`quality=101` selects lossless mode in OpenCV's encoder).
- `preprocess_dataset.py` now refuses to write a task's index if any shard save failed,
  so interrupted preprocessing runs are safely resumable by re-running.
