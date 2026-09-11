import laspy
import numpy as np
import pytest
from s20_pipeline.desktop import preview
from s20_pipeline.storage import digest


def test_preview_is_bounded_preserves_source_and_world_coordinates(tmp_path):
    source = tmp_path / "capture.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.array([1_000_000, 2_000_000, 0])
    header.scales = np.full(3, 0.0001)
    data = laspy.LasData(header)
    n = 30003
    data.x = 1_000_000 + np.arange(n) * 0.0001
    data.y = np.full(n, 2_000_000.0)
    data.z = np.zeros(n)
    data.red = np.full(n, 65535, dtype="u2")
    data.green = np.zeros(n, dtype="u2")
    data.blue = np.full(n, 32768, dtype="u2")
    data.write(source)
    before = digest(source)
    value = preview(source, tmp_path / "cache", 10000)
    points = np.fromfile(value["file"], dtype="f4").reshape(-1, 6)
    assert len(points) <= 10000 and len(points) == value["display_points"]
    assert points.nbytes == value["bytes"]
    assert np.max(abs(points[:, :3] + value["origin"] - data.xyz[::4])) < 1e-6
    assert np.all(points[:, 3] == 1) and np.all(points[:, 4] == 0)
    assert preview(source, tmp_path / "cache", 10000) == value
    assert digest(source) == before
    with pytest.raises(ValueError):
        preview(source, tmp_path / "cache", 2000001)
