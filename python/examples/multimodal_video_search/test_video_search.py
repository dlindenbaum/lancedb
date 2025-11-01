"""
Unit tests for video search functionality

Run with: pytest test_video_search.py
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
from pathlib import Path

# Skip tests if dependencies not available
pytest.importorskip("torch")
pytest.importorskip("open_clip")
pytest.importorskip("lancedb")


from video_search import (
    HybridVideoSearch,
    DINOv3Embeddings,
    DenseFeatureExtractor,
    VideoFrameExtractor
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image"""
    from PIL import Image

    img = Image.new('RGB', (224, 224), color='red')
    img_path = os.path.join(temp_dir, "test_image.jpg")
    img.save(img_path)
    return img_path


@pytest.fixture
def search_system(temp_dir):
    """Create a test search system"""
    db_path = os.path.join(temp_dir, "test.db")
    search = HybridVideoSearch(
        db_path=db_path,
        device="cpu",  # Use CPU for tests
        use_dino=False,  # Disable DINOv3 for faster tests
        batch_size=2
    )
    return search


class TestDINOv3Embeddings:
    """Test DINOv3 embedding extraction"""

    def test_init(self):
        """Test DINOv3 initialization"""
        embedder = DINOv3Embeddings(device="cpu")
        assert embedder.device == "cpu"
        assert embedder.model is not None

    def test_embed_image(self, sample_image):
        """Test single image embedding"""
        embedder = DINOv3Embeddings(device="cpu")
        embedding = embedder.embed_image(sample_image)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # Should be 1D vector
        assert embedding.shape[0] > 0  # Should have features

        # Check normalization
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5  # Should be normalized

    def test_embed_batch(self, sample_image):
        """Test batch embedding"""
        embedder = DINOv3Embeddings(device="cpu")
        embeddings = embedder.embed_batch([sample_image, sample_image], batch_size=2)

        assert len(embeddings) == 2
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)


class TestDenseFeatureExtractor:
    """Test dense feature extraction"""

    def test_extract_dense_clip(self, sample_image):
        """Test dense CLIP feature extraction"""
        extractor = DenseFeatureExtractor(device="cpu")
        features, grid_size = extractor.extract_dense_clip(sample_image)

        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 3  # (grid_size, grid_size, feature_dim)
        assert features.shape[0] == grid_size
        assert features.shape[1] == grid_size

    def test_extract_text_features(self):
        """Test text feature extraction"""
        extractor = DenseFeatureExtractor(device="cpu")
        features = extractor.extract_text_features("test query")

        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 1  # Should be 1D vector

    def test_compute_similarity_map(self, sample_image):
        """Test similarity map computation"""
        extractor = DenseFeatureExtractor(device="cpu")
        features, grid_size = extractor.extract_dense_clip(sample_image)
        similarity_map = extractor.compute_similarity_map(features, "red color")

        assert isinstance(similarity_map, np.ndarray)
        assert similarity_map.shape == (grid_size, grid_size)
        assert np.all(similarity_map >= -1) and np.all(similarity_map <= 1)


class TestVideoFrameExtractor:
    """Test video frame extraction"""

    def test_extract_frames_no_video(self, temp_dir):
        """Test handling of non-existent video"""
        with pytest.raises(Exception):
            VideoFrameExtractor.extract_frames(
                "nonexistent.mp4",
                temp_dir,
                fps=1
            )


class TestHybridVideoSearch:
    """Test main search system"""

    def test_init(self, temp_dir):
        """Test initialization"""
        db_path = os.path.join(temp_dir, "test.db")
        search = HybridVideoSearch(db_path=db_path, device="cpu", use_dino=False)

        assert search.db_path == db_path
        assert search.device == "cpu"
        assert not search.use_dino

    def test_search_text_empty_db(self, search_system):
        """Test text search on empty database"""
        # Should handle empty database gracefully
        with pytest.raises(Exception):
            # Table doesn't exist yet
            search_system.search_text("test query")

    def test_locate_object(self, search_system, sample_image):
        """Test object localization"""
        detections = search_system.locate_object(
            sample_image,
            query="red object",
            threshold=0.1
        )

        assert isinstance(detections, list)
        # May or may not find detections depending on threshold

    def test_nms(self, search_system):
        """Test Non-Maximum Suppression"""
        detections = [
            {"bbox": (10, 10, 50, 50), "score": 0.9},
            {"bbox": (15, 15, 50, 50), "score": 0.8},  # Overlaps with first
            {"bbox": (100, 100, 50, 50), "score": 0.7}  # Different location
        ]

        filtered = search_system._nms(detections, iou_threshold=0.5)

        # Should keep first and third (second overlaps with first)
        assert len(filtered) <= len(detections)
        assert filtered[0]["score"] >= filtered[-1]["score"]  # Sorted by score

    def test_compute_iou(self, search_system):
        """Test IoU computation"""
        # Perfect overlap
        iou = search_system._compute_iou((0, 0, 10, 10), (0, 0, 10, 10))
        assert abs(iou - 1.0) < 1e-5

        # No overlap
        iou = search_system._compute_iou((0, 0, 10, 10), (20, 20, 10, 10))
        assert abs(iou - 0.0) < 1e-5

        # Partial overlap
        iou = search_system._compute_iou((0, 0, 10, 10), (5, 5, 10, 10))
        assert 0.0 < iou < 1.0

    def test_get_stats_empty(self, search_system):
        """Test statistics on empty database"""
        stats = search_system.get_stats()

        assert isinstance(stats, dict)
        assert "tables" in stats
        assert "total_frames" in stats
        assert stats["total_frames"] == 0

    def test_visualize_detections(self, search_system, sample_image, temp_dir):
        """Test detection visualization"""
        detections = [
            {"bbox": (10, 10, 50, 50), "score": 0.9}
        ]

        output_path = os.path.join(temp_dir, "viz.jpg")
        result_img = search_system.visualize_detections(
            sample_image,
            detections,
            output_path=output_path
        )

        assert result_img is not None
        assert os.path.exists(output_path)


class TestIntegration:
    """Integration tests"""

    def test_full_workflow_with_mock_data(self, temp_dir, sample_image):
        """Test complete workflow with sample data"""
        # Initialize
        db_path = os.path.join(temp_dir, "integration.db")
        search = HybridVideoSearch(
            db_path=db_path,
            device="cpu",
            use_dino=False,
            batch_size=2
        )

        # Create mock frame data
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir)

        # Copy sample image multiple times
        frame_paths = []
        for i in range(3):
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.jpg")
            shutil.copy(sample_image, frame_path)
            frame_paths.append(frame_path)

        # Manually create table with mock data
        import lancedb
        from lancedb.embeddings import OpenClipEmbeddings

        clip_embeddings = OpenClipEmbeddings(device="cpu")

        data = []
        for i, frame_path in enumerate(frame_paths):
            embedding = clip_embeddings.generate_image_embedding(frame_path)
            data.append({
                "video_id": "test_video",
                "frame_id": i,
                "frame_path": frame_path,
                "timestamp": float(i * 0.5),
                "frame_number": i,
                "clip_embedding": embedding
            })

        db = lancedb.connect(db_path)
        db.create_table("clip_embeddings", data)

        # Test text search
        results = search.search_text("test query", limit=2)
        assert len(results) <= 2
        assert all("frame_id" in r for r in results)

        # Test stats
        stats = search.get_stats()
        assert stats["total_frames"] == 3
        assert "test_video" in stats["videos"]


def test_imports():
    """Test that all modules can be imported"""
    from video_search import (
        HybridVideoSearch,
        DINOv3Embeddings,
        DenseFeatureExtractor,
        VideoFrameExtractor
    )

    assert HybridVideoSearch is not None
    assert DINOv3Embeddings is not None
    assert DenseFeatureExtractor is not None
    assert VideoFrameExtractor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
