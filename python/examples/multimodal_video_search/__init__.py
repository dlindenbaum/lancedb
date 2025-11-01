"""
Multimodal Video Search with CLIP + DINOv3 + Face Recognition + LanceDB

A comprehensive video search system combining semantic and visual understanding
with face detection and recognition capabilities.
"""

from .video_search import (
    HybridVideoSearch,
    DINOv3Embeddings,
    DenseFeatureExtractor,
    VideoFrameExtractor
)

# Face detection imports (optional)
try:
    from .face_embeddings import (
        FaceEmbeddings,
        FaceDatabase
    )
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FaceEmbeddings = None
    FaceDatabase = None
    FACE_DETECTION_AVAILABLE = False

__version__ = "0.2.0"
__all__ = [
    "HybridVideoSearch",
    "DINOv3Embeddings",
    "DenseFeatureExtractor",
    "VideoFrameExtractor",
    "FaceEmbeddings",
    "FaceDatabase",
    "FACE_DETECTION_AVAILABLE"
]
