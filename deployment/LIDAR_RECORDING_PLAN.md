# Lidar Recording Plan (Raspberry Pi -> offline SLAM on Mac)

Status: design only, no code written. Target: Raspberry Pi OS 13 (trixie), aarch64, Python 3.13.

## 1. Goals

- Record raw 2D lidar revolutions on the Pi from `deployment/src/visualize_lidar_live.py` (pyserial, no ROS) as a new `--record` mode.
- Store them losslessly (raw units), crash-safe, and cheap on the SD card, so `build_map.py` on the Mac can run scan-matching SLAM offline.
- No IMU or wheel odometry exists, so SLAM must rely on scan matching alone. Timestamps and clean revolutions matter more than anything else.
- Non-goals: live SLAM on the Pi, compression, ROS bag compatibility.

**Caveat: the parser is unverified.** `crc8` (poly 0x4D), the field offsets (start angle at bytes 4-5, end angle at 42-43, points from byte 6, 3 bytes each), the 47-byte length and the 230400 baud were written from memory. They have not been checked against a real device. Step 0 below must pass before any recording is trusted.

### Step 0: verification (before building anything)
1. `python3 visualize_lidar_live.py --port /dev/ttyUSB0 --raw` should show `54 2c` repeating at 47-byte spacing.
2. Run the `--out snap.png` mode for 5 s. Expect `bad` near 0 and `ok` around 4000-5000 packets over 5 s... more precisely about 10 rev/s x ~38 packets/rev (~450 points / 12) = roughly 380 packets/s. If `bad` is a large fraction, the CRC, the baud or the model is wrong.
3. Sanity check geometry: put the lidar in a rectangular room or near a wall and check that the PNG shows straight walls. Rotate or move a hand at a known angle to confirm angle direction (decides `--flip`).
4. Compare with the spec or datasheet for the actual model (LD19 vs LD06 vs MS200). Check the packet also carries a speed field (bytes 2-3, deg/s) and a device timestamp (bytes 44-45, ms, wraps at 30000). Consider logging both.

## 2. `--record` mode

New flags (all optional, defaults in brackets):
- `--record DIR` : enable recording; session directory is created under DIR. No window and no PNG in this mode.
- `--port` [/dev/ttyUSB0 for this Pi; the existing `/dev/ldlidar` default stays], `--baud` [230400], `--flip`
- `--duration SEC` [0 = unlimited] ; `--max-frames N` [0 = unlimited]
- `--min-frames-warn` / `--chunk-seconds SEC` [30] : chunk rollover (see 5)
- `--label TEXT` : free-text note stored in metadata (e.g. "hallway loop 1")

Stop conditions: duration reached, max-frames reached, Ctrl-C / SIGTERM (handled so the last chunk and metadata are finalized), serial error or no valid packet for 5 s (log, finalize, exit non-zero), free disk below 200 MB (finalize and exit).

## 3. What is a frame

One frame = one full revolution (about 360 deg, 450 points at ~10 Hz).
- Per point angle (centi-degrees, 0-35999) is computed as in `parse_packet` (linear interpolation between start and end angle of the packet, with wrap at 360).
- Wraparound detection is on the packet's start angle: if `start_angle < previous packet start_angle` (with a margin, e.g. drop of more than 180 deg), the previous revolution is complete and a new frame begins at that packet. Do this per packet, not per point, so a frame holds whole packets (the packet that straddles 360 deg goes into the new frame, its points before wrap are kept with the angle value as is).
- First, partial revolution: discard everything before the first wrap (never recorded).
- Last, partial revolution: discarded on shutdown, but counted in metadata (`dropped_partial_frames`).
- Gaps: if a frame has fewer than ~80% of the median point count or the sum of angular gaps between successive packets exceeds 10 deg, still store it but set `flags` (see 4). Do not interpolate or drop silently. Bad-CRC packets inside a frame leave an angular hole, and that is recorded via `crc_bad`.
- `--flip` is not applied to stored data. Raw angles are stored, and `flip` is metadata, applied by the loader. That way a wrong flip guess can be fixed offline.

## 4. Per-frame data

Stored raw, no float conversion on the Pi:

| Field | Type | Notes |
|---|---|---|
| `frame_id` | uint32 | monotonic from 0 |
| `t_mono` | float64 | `time.monotonic()` at completion of the first packet of the frame. Also a per-packet `pkt_t_mono` array (float64) is stored since a revolution takes ~100 ms and SLAM deskewing may use it |
| `t_wall` | float64 | `time.time()` at the same instant, used only for human alignment |
| `angle_cdeg` | uint16[N] | centi-degrees, 0-35999 |
| `dist_mm` | uint16[N] | 0 means invalid |
| `intensity` | uint8[N] | |
| `pkt_idx` | uint16[N] | which packet each point came from (index into `pkt_t_mono`) |
| `n_packets`, `crc_bad` | uint16 | good packets in this frame, CRC failures seen since previous frame |
| `resync_bytes` | uint32 | bytes discarded while resyncing |
| `flags` | uint8 | bit0 low point count, bit1 angular gap, bit2 timing anomaly |

Host timestamps carry USB/serial buffering latency (jitter of several ms), which is acceptable. Optionally log the device timestamp field per packet if Step 0 confirms it.

## 5. Storage format

| Option | Crash safety | Append | Size (30 min, 18000 frames, ~9 B/pt) | Verdict |
|---|---|---|---|---|
| One `.npz` per session (written at end) | Poor, lose all on crash or power cut | No | ~75 MB raw | No |
| JSONL per frame | Good | Yes | ~5x larger (~400 MB), slow parse | No |
| One file per frame | Good | Yes | Same bytes, but 18000 files, SD metadata overhead | No |
| **Chunked `.npz` (uncompressed) + `meta.json`** | Good, lose at most one chunk | Yes (new chunk file) | ~75 MB | **Recommended** |

Recommendation: chunks of about 30 s (~300 frames, ~1.2 MB). Each chunk is one `np.savez` (not compressed, to save CPU) of ragged data stored as flat concatenated arrays plus `frame_offsets` (int64, len = frames+1) and per-frame arrays. Write to `chunk_NNNN.npz.tmp`, `fsync`, then `os.replace` to `chunk_NNNN.npz`, so a reader never sees a half-written chunk. Loader ignores `*.tmp`.

Layout:
```
recordings/
  2026-10-07_153012_hallway/        # start time (local) + label
    meta.json                       # written at start, rewritten (atomic) at end
    chunk_0000.npz
    chunk_0001.npz
    ...
```
Session names never collide (second resolution plus label); refuse to overwrite.

`meta.json` header: schema_version, port, baud, lidar model guess ("LD19-compatible, unverified"), packet format constants (header, length, points per packet), flip flag, script name and `git rev-parse --short HEAD` (plus a `dirty` flag from `git status --porcelain`), start time (wall, ISO 8601, and monotonic at start), hostname, Python and pyserial and numpy versions, label, CLI args. At the end add: end time, frames written, chunks, total good/bad packets, dropped partial frames, stop reason.

## 6. Write strategy

- Serial read loop must never block on disk. Use a reader thread/main loop that assembles frames and pushes finished chunks (or frames) into a bounded `queue.Queue`; a writer thread does `np.savez` + fsync. If the queue fills, drop and count (`dropped_chunks`) rather than stall the serial read. 230400 baud is ~23 KB/s, and the kernel buffer holds only ~4 KB on USB-serial, so a stall of more than ~100 ms loses data.
- Buffer frames in RAM per chunk (about 1.2 MB). Flush and fsync once per chunk (every ~30 s), not per frame. Max loss on power cut: 30 s.
- SD wear: ~75 MB per 30 min with one fsync per 30 s is negligible. Prefer a USB stick or tmpfs plus copy if recording many hours.
- Update `meta.json` only at start and end (and optionally at each chunk, atomic replace).
- The record path must also avoid per-point Python work where possible (collect bytes, decode with numpy `frombuffer` at chunk time). Pure Python at 4500 pts/s is fine on a Pi 4/5 but should be measured.

## 7. Validation and sanity checks

Live (printed once per second, to stderr): frames, rev rate, good/bad packets, queue depth.
At the end and in the offline loader:
- Rev rate within 8-12 Hz; frame time deltas without gaps greater than 3x median.
- Median points per frame consistent; CRC bad ratio below 1%; angle coverage per frame at least 95%.
- Valid points (dist > 0) at least 50% of the total, and distances within the model's range (e.g. 0.02-12 m).
- Monotonic `t_mono` and increasing `frame_id`; chunk frame_ids contiguous across chunks.
- Print a verdict (OK / WARN) and write it into `meta.json`.

## 8. Transfer to the Mac

```bash
# from the Mac, one session (resumable, only changed files)
rsync -avP pi@raspberrypi.local:~/recordings/2026-10-07_153012_hallway/ \
      ~/Documents/gi_work_trial/lidar_recordings/2026-10-07_153012_hallway/

# or whole tree
rsync -avP pi@raspberrypi.local:~/recordings/ ~/Documents/gi_work_trial/lidar_recordings/

# one-off
scp -r pi@raspberrypi.local:~/recordings/<session> ./
```
Do not copy a session that is still being recorded without checking that `meta.json` has an end time (or tolerate a missing one, since the chunks are still valid). Hostname and user are assumptions to adjust.

## 9. Offline loader interface (for `build_map.py`)

A small module, e.g. `lidar_io.py`, with no Pi dependencies (numpy only):
- `load_session(path, flip=None) -> Session` : reads `meta.json` and all chunks in order, ignoring `.tmp` files and tolerating a missing end-of-run metadata.
- `Session.meta` : dict. `Session.num_frames`.
- `Session.frame(i) -> Frame` / iteration, where `Frame` has `.t_mono`, `.t_wall`, `.angles` (rad, float32, flip applied, CCW, 0 = forward), `.ranges` (m, float32, invalid = NaN), `.intensity` (uint8), `.pkt_t` (per-point timestamps), `.flags`, `.crc_bad`.
- `Frame.xy(min_range=0.05, max_range=12.0, min_intensity=0) -> (M, 2) float32` : Cartesian points in the sensor frame, x forward, y left (same convention as the live viewer).
- `Session.timestamps()` : array of `t_mono` for sync and rate checks; `Session.summary()` prints the section 7 checks.
Convention to fix now: SI units (m, rad, s) after loading, raw integer units on disk.

## 10. Risks and open questions

- Parser unverified (see Step 0). The wrong model or baud shows up as a high CRC bad rate.
- Port naming: `/dev/ttyUSB0` may change after a replug; consider a udev rule or `/dev/serial/by-id/...` and make the port a required flag in record mode.
- Permissions: user must be in `dialout`.
- Angle direction and zero offset are unknown (`--flip`, mounting angle) and affect SLAM. Mounting offset is not recorded, so add `--mount-yaw-deg` to metadata if known.
- Motion distortion within one revolution (no odometry) degrades scan matching, so move slowly. Per-packet timestamps allow deskewing later.
- Single-beam 2D in featureless corridors causes SLAM drift (degenerate along the corridor axis). Plan loops or distinctive features.
- Host timestamp jitter vs the device timestamp; is the device timestamp worth storing?
- Does the real lidar report 10 Hz and about 450 points per revolution? Sizes above assume so.
- Pi power: USB lidar plus motors browning out the supply can cause resets, so rely on chunking.
- Open: keep the unused dist 0 points or drop them? Proposed: keep, they are tiny, and let the loader mask them.

## 11. Implementation checklist

1. Run Step 0 verification on the Pi; fix parser (CRC, offsets, baud, angle direction) if needed.
2. Add `--record`, `--duration`, `--max-frames`, `--chunk-seconds`, `--label` flags; make `--port` explicit for record mode.
3. Refactor frame assembly (packet-level wraparound detection, discard first partial) into a function independent of OpenCV; make `cv2` import lazy so recording works headless.
4. Add session dir creation, `meta.json` writer (git commit, args, versions) with atomic replace.
5. Add writer thread, bounded queue, chunk `.npz` + fsync + rename.
6. Add signal handling, stop conditions, final metadata, and the end-of-run validation report.
7. Record a 60 s test on the Pi, inspect chunk contents and sanity checks.
8. Write `lidar_io.py` loader and a tiny round-trip check on the Mac (rsync the test session, plot one frame).
9. Do the 30 min recording, rsync, and hand off to `build_map.py`.
