# T19 - LiDAR Viewer and Orange Trim

DEPENDS ON: real Stray Scanner-style capture in `Lidar/75b9d04d0a`
CREATE: `tools/lidar_viewer_server.py`, `tools/lidar_viewer.html`

## Purpose

Provide a local browser viewer for the iPhone LiDAR export and a quick way to trim the
capture to the frames containing only the oranges.

## Localhost

Run from the project root:

```bash
python tools/lidar_viewer_server.py
```

Open:

```text
http://127.0.0.1:8765
```

## Input Capture

```text
Lidar/75b9d04d0a/
  depth/
  confidence/
  camera_matrix.csv
  odometry.csv
  imu.csv
  rgb.mp4
```

Verified capture contents:

- `2110` depth PNG frames
- `2110` confidence PNG frames
- RGB video is `1920 x 1440`, `60 fps`, about `35.15 s`
- Depth frames are `256 x 192`, 16-bit PNG
- Confidence frames are `256 x 192`, 8-bit PNG

## Orange Trim Workflow

1. Open the local viewer.
2. Scrub the RGB video until the orange-only section starts.
3. Copy the displayed frame number into `Start frame`.
4. Scrub until the orange-only section ends.
5. Copy the displayed frame number into `End frame`.
6. Click `Create Trim`.

The server creates:

```text
Lidar/trims/oranges_START_END/
  depth/
  confidence/
  camera_matrix.csv
  odometry.csv
  rgb.mp4
```

## Notes

- The trim keeps synchronized depth, confidence, odometry, calibration, and RGB video.
- The output frames are renumbered from `000000`.
- `camera_matrix.csv` is copied unchanged.
- `odometry.csv` is filtered to the selected frame range and renumbered.
