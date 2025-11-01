"""
Multimodal Video Search with CLIP + DINOv3 + LanceDB

A comprehensive video search system combining semantic and visual understanding.
"""

from .video_search import (
    HybridVideoSearch,
    DINOv3Embeddings,
    DenseFeatureExtractor,
    VideoFrameExtractor
)

__version__ = "0.1.0"
__all__ = [
    "HybridVideoSearch",
    "DINOv3Embeddings",
    "DenseFeatureExtractor",
    "VideoFrameExtractor"
]
