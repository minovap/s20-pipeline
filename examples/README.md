# Invocation examples

Use new output directories and absolute source paths. Capture/model/output assets are intentionally not checked in.

```sh
# Complete supported raw pipeline; explicit calibration/clock contract.
s20 run /Volumes/SD_CARD/Test --output /data/results/test \
  --camera-clock sensor-header --camera-convention lidar-extrinsics \
  --resources throughput --memory-gb 16

# Geometry only, retaining all available frames.
s20 run /data/capture --output /data/results/geometry --no-color

# Local exposure + visibility consensus, with precomputed person masks.
s20 colorize --geometry /data/geometry.ply --cameras /data/cameras.json \
  --calibration /data/calibration.yaml --masks /data/masks \
  --output /results/color --resources throughput

# Same color method through the CPU reference; useful for GPU parity checks.
s20 colorize --geometry /data/geometry.ply --cameras /data/cameras.json \
  --calibration /data/calibration.yaml --masks /data/masks \
  --output /results/cpu-color --blend cpu
```
