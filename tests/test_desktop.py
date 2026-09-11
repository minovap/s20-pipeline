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


def test_export_slices_writes_union_once_with_colors(tmp_path):
    from s20_pipeline.desktop import export_slices

    source = tmp_path / "cloud.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.array([100.0, 200.0, 0.0])
    header.scales = np.full(3, 0.001)
    data = laspy.LasData(header)
    grid = np.arange(10, dtype=float)
    data.x = 100 + grid
    data.y = np.full(10, 200.0)
    data.z = np.zeros(10)
    data.red = np.full(10, 65535, dtype="u2")
    data.green = np.arange(10, dtype="u2") * 1000
    data.blue = np.zeros(10, dtype="u2")
    data.write(source)
    before = digest(source)
    # Two overlapping boxes: x in [102,105] and x in [104,107] -> points 102..107 = 6 unique points.
    spec = {
        "output": str(tmp_path / "out" / "slice.las"),
        "sources": [
            {
                "path": str(source),
                "boxes": [[[102, 199, -1], [105, 201, 1]], [[104, 199, -1], [107, 201, 1]]],
            }
        ],
    }
    result = export_slices(spec)
    assert result["points"] == 6
    out = laspy.read(spec["output"])
    assert sorted(np.asarray(out.x).tolist()) == [102, 103, 104, 105, 106, 107]
    assert out.red[0] == 65535
    assert set(np.asarray(out.green).tolist()) == {2000, 3000, 4000, 5000, 6000, 7000}
    assert digest(source) == before
    with pytest.raises(FileExistsError):
        export_slices(spec)
    with pytest.raises(ValueError):
        export_slices({**spec, "output": str(tmp_path / "none.las"), "sources": [{"path": str(source), "boxes": [[[900, 900, 900], [901, 901, 901]]]}]})


def test_export_slices_applies_orientation_transform(tmp_path):
    from s20_pipeline.desktop import export_slices

    source = tmp_path / "tilted.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.full(3, 0.001)
    data = laspy.LasData(header)
    data.x = np.array([0.0, 10.0, 20.0])
    data.y = np.zeros(3)
    data.z = np.zeros(3)
    data.red = data.green = data.blue = np.zeros(3, dtype="u2")
    data.write(source)
    # Rotate 90 degrees about z around origin (10,0,0): x -> y. Point at x=20 lands at y=10.
    rot = [0, -1, 0, 1, 0, 0, 0, 0, 1]
    spec = {
        "output": str(tmp_path / "rotated.las"),
        "sources": [{"path": str(source), "boxes": [[[9, 9, -1], [11, 11, 1]]], "transform": {"rotation": rot, "origin": [10, 0, 0], "translation": [0, 0, 0.5]}}],
    }
    result = export_slices(spec)
    out = laspy.read(spec["output"])
    assert result["points"] == 1
    assert np.allclose([out.x[0], out.y[0], out.z[0]], [10, 10, 0.5], atol=1e-3)
