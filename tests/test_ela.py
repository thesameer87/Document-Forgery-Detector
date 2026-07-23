# tests/test_ela.py
"""
Minimum viable tests for the ELA preprocessing pipeline.

All tests use synthetic images — no dependency on CASIA v2 data.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.ela import batch_process_ela, compute_ela


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solid_image() -> Image.Image:
    """A 100x100 solid-colour RGB image (deterministic)."""
    arr = np.full((100, 100, 3), fill_value=128, dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def gradient_image() -> Image.Image:
    """A 100x200 RGB image with a horizontal gradient (more ELA variance)."""
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    for c in range(3):
        arr[:, :, c] = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (100, 1))
    return Image.fromarray(arr)


@pytest.fixture
def tmp_image_path(solid_image, tmp_path) -> Path:
    """Save the solid image to a temp file and return its path."""
    path = tmp_path / "test_image.jpg"
    solid_image.save(str(path), format="JPEG", quality=95)
    return path


@pytest.fixture
def mock_dataset(tmp_path) -> tuple:
    """Create a minimal CASIA-like directory structure with synthetic images.

    Returns:
        (raw_dir, ela_dir, config)
    """
    raw_dir = tmp_path / "raw"
    ela_dir = tmp_path / "ela"

    # Create Au/ and Tp/ sub-folders with a few images each
    for folder, fill_val in [("Au", 100), ("Tp", 200)]:
        d = raw_dir / folder
        d.mkdir(parents=True)
        for i in range(3):
            arr = np.full((50, 50, 3), fill_value=fill_val + i * 10, dtype=np.uint8)
            Image.fromarray(arr).save(str(d / f"img_{i:03d}.jpg"), format="JPEG")

    config = {
        "dataset": {
            "root_dir": str(raw_dir),
            "ela_dir": str(ela_dir),
            "image_extensions": [".jpg", ".png"],
        },
        "ela": {
            "quality": 90,
            "scale": 15,
        },
    }
    return raw_dir, ela_dir, config


# ---------------------------------------------------------------------------
# Tests: compute_ela
# ---------------------------------------------------------------------------

class TestComputeEla:
    """Unit tests for the core compute_ela function."""

    def test_output_shape_matches_input(self, tmp_image_path):
        """ELA output must have the same (H, W, 3) shape as the input."""
        original = Image.open(tmp_image_path)
        w, h = original.size
        ela = compute_ela(tmp_image_path)
        assert ela.shape == (h, w, 3)

    def test_output_dtype_is_uint8(self, tmp_image_path):
        """ELA map must be uint8 (0-255)."""
        ela = compute_ela(tmp_image_path)
        assert ela.dtype == np.uint8

    def test_output_values_in_range(self, tmp_image_path):
        """All pixel values must be in [0, 255]."""
        ela = compute_ela(tmp_image_path)
        assert ela.min() >= 0
        assert ela.max() <= 255

    def test_accepts_pil_image(self, solid_image):
        """compute_ela should accept a PIL Image directly."""
        ela = compute_ela(solid_image)
        assert ela.shape == (100, 100, 3)

    def test_accepts_numpy_array(self, solid_image):
        """compute_ela should accept a NumPy array."""
        arr = np.array(solid_image)
        ela = compute_ela(arr)
        assert ela.shape == (100, 100, 3)

    def test_different_quality_produces_different_output(self, gradient_image):
        """Different JPEG quality levels should yield different ELA maps."""
        ela_q90 = compute_ela(gradient_image, quality=90, scale=15)
        ela_q30 = compute_ela(gradient_image, quality=30, scale=15)
        # Lower quality → larger re-compression artefacts → higher ELA values
        assert not np.array_equal(ela_q90, ela_q30)
        assert ela_q30.mean() > ela_q90.mean()

    def test_scale_amplifies_output(self, gradient_image):
        """Higher scale should produce brighter (higher mean) ELA maps."""
        ela_s1 = compute_ela(gradient_image, quality=90, scale=1)
        ela_s50 = compute_ela(gradient_image, quality=90, scale=50)
        assert ela_s50.mean() >= ela_s1.mean()

    def test_invalid_quality_raises(self, solid_image):
        """quality outside 1-100 must raise ValueError."""
        with pytest.raises(ValueError):
            compute_ela(solid_image, quality=0)
        with pytest.raises(ValueError):
            compute_ela(solid_image, quality=101)

    def test_missing_file_raises(self):
        """Non-existent file path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_ela("nonexistent_image.jpg")

    def test_unsupported_type_raises(self):
        """Passing an unsupported type must raise TypeError."""
        with pytest.raises(TypeError):
            compute_ela(12345)


# ---------------------------------------------------------------------------
# Tests: batch_process_ela
# ---------------------------------------------------------------------------

class TestBatchProcessEla:
    """Integration tests for batch ELA processing."""

    def test_processes_all_images(self, mock_dataset):
        """All images should be processed and stats should be correct."""
        raw_dir, ela_dir, config = mock_dataset
        stats = batch_process_ela(config)
        assert stats["processed"] == 6  # 3 Au + 3 Tp
        assert stats["skipped"] == 0
        assert stats["errors"] == 0

    def test_preserves_directory_structure(self, mock_dataset):
        """Output must mirror the input directory structure (Au/, Tp/)."""
        raw_dir, ela_dir, config = mock_dataset
        batch_process_ela(config)
        assert (ela_dir / "Au").is_dir()
        assert (ela_dir / "Tp").is_dir()

    def test_output_files_are_png(self, mock_dataset):
        """All ELA outputs should be saved as .png files."""
        _, ela_dir, config = mock_dataset
        batch_process_ela(config)
        for f in ela_dir.rglob("*"):
            if f.is_file():
                assert f.suffix == ".png"

    def test_skip_existing(self, mock_dataset):
        """Already-processed images should be skipped on a second run."""
        _, ela_dir, config = mock_dataset
        batch_process_ela(config)
        stats = batch_process_ela(config)
        assert stats["processed"] == 0
        assert stats["skipped"] == 6

    def test_force_reprocesses(self, mock_dataset):
        """With force=True, existing outputs should be overwritten."""
        _, ela_dir, config = mock_dataset
        batch_process_ela(config)
        stats = batch_process_ela(config, force=True)
        assert stats["processed"] == 6
        assert stats["skipped"] == 0

    def test_empty_raw_dir(self, tmp_path):
        """An empty (but existing) raw dir should return zero counts."""
        raw_dir = tmp_path / "raw_empty"
        ela_dir = tmp_path / "ela_empty"
        raw_dir.mkdir()
        config = {
            "dataset": {
                "root_dir": str(raw_dir),
                "ela_dir": str(ela_dir),
                "image_extensions": [".jpg"],
            },
            "ela": {"quality": 90, "scale": 15},
        }
        stats = batch_process_ela(config)
        assert stats["processed"] == 0

    def test_missing_raw_dir(self, tmp_path):
        """A non-existent raw dir should warn and return zero counts."""
        config = {
            "dataset": {
                "root_dir": str(tmp_path / "does_not_exist"),
                "ela_dir": str(tmp_path / "ela"),
                "image_extensions": [".jpg"],
            },
            "ela": {"quality": 90, "scale": 15},
        }
        stats = batch_process_ela(config)
        assert stats["processed"] == 0
