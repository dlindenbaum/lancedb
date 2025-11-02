"""
Hybrid Video Search System using ONNX Models + LanceDB

This module provides a complete multimodal video search system that combines:
- CLIP (ONNX) for semantic text-to-image search
- DINOv3 (ONNX) for fine-grained visual features
- Face detection and recognition (DeepFace)
- LanceDB for efficient vector storage and retrieval

All models use ONNX for efficient, lightweight inference without PyTorch dependency.
"""

import io
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# LanceDB imports
import lancedb
import pyarrow as pa

# ONNX-based embeddings
try:
    from lancedb_onnx_embeddings import create_onnx_clip_embeddings
    from onnx_embeddings import ONNXDINOv3Embeddings, ONNXCLIPEmbeddings
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX models not available. Install with: pip install onnxruntime transformers")

# Face detection imports (optional)
try:
    from face_embeddings import FaceEmbeddings, FaceDatabase
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    FaceEmbeddings = None
    FaceDatabase = None


class DINOv3Embeddings:
    """
    DINOv3 embedding function using ONNX for efficient inference

    Wrapper around ONNXDINOv3Embeddings for compatibility
    """

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        """
        Initialize DINOv3 ONNX model

        Args:
            model_name: DINOv3 model variant (dinov2_vits14, dinov2_vitb14)
            device: Device to run model on ("cuda" or "cpu")
        """
        import onnxruntime as ort

        self.device = device
        self.model_name = model_name

        # Determine ONNX providers
        if device == "cuda":
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print("⚠ CUDA requested but not available, using CPU")
                providers = ['CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Load ONNX model
        self.onnx_model = ONNXDINOv3Embeddings(
            model_name=model_name,
            providers=providers
        )

    def embed_image(self, image: Union[str, bytes, Image.Image]) -> np.ndarray:
        """
        Extract DINOv3 embedding for a single image

        Args:
            image: Image path, bytes, or PIL Image

        Returns:
            Normalized embedding vector
        """
        # Handle bytes
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        return self.onnx_model.encode_image(image)

    def embed_batch(self, images: List[Union[str, bytes, Image.Image]],
                   batch_size: int = 32) -> List[np.ndarray]:
        """
        Extract DINOv3 embeddings for a batch of images

        Args:
            images: List of image paths, bytes, or PIL Images
            batch_size: Number of images to process at once

        Returns:
            List of normalized embedding vectors
        """
        # Convert bytes to PIL Images
        processed_images = []
        for img in images:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            processed_images.append(img)

        return self.onnx_model.encode_images_batch(processed_images, batch_size)

    def extract_dense_features(self, image: Union[str, Image.Image]) -> Tuple[np.ndarray, int]:
        """
        Extract dense patch-level features for object localization

        Args:
            image: Image path or PIL Image

        Returns:
            Tuple of (features_grid, grid_size)
            features_grid: (grid_size, grid_size, feature_dim)
            grid_size: Number of patches per dimension
        """
        return self.onnx_model.extract_dense_features(image)


class DenseFeatureExtractor:
    """
    Extract dense patch-level features using ONNX models for object localization

    Note: For dense features, we use DINOv3 which naturally provides patch features
    """

    def __init__(self, clip_model: str = "clip_vit_b32", device: str = "cuda"):
        """
        Initialize dense feature extractor

        Args:
            clip_model: CLIP model variant (for text encoding)
            device: Device to run model on
        """
        import onnxruntime as ort

        self.device = device

        # Determine ONNX providers
        if device == "cuda":
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Use CLIP ONNX for text encoding
        self.clip_model = ONNXCLIPEmbeddings(
            model_name=clip_model,
            providers=providers
        )

        # Use DINOv3 ONNX for dense features (better for localization)
        self.dino_model = ONNXDINOv3Embeddings(
            model_name="dinov2_vitb14",
            providers=providers
        )

    def extract_dense_clip(self, image: Union[str, Image.Image]) -> Tuple[np.ndarray, int]:
        """
        Extract dense features using DINOv3 (better for localization than CLIP)

        Args:
            image: Image path or PIL Image

        Returns:
            Tuple of (features_grid, grid_size)
        """
        # Use DINOv3 which naturally provides patch-level features
        return self.dino_model.extract_dense_features(image)

    def extract_text_features(self, text: str) -> np.ndarray:
        """
        Extract CLIP text features using ONNX

        Args:
            text: Text query

        Returns:
            Normalized text embedding
        """
        return self.clip_model.encode_text(text)

    def compute_similarity_map(self, patch_features: np.ndarray,
                              text_query: str) -> np.ndarray:
        """
        Compute similarity map between image patches and text query

        Args:
            patch_features: Dense patch features (grid_size, grid_size, feature_dim)
            text_query: Text description

        Returns:
            Similarity map (grid_size, grid_size)
        """
        text_features = self.extract_text_features(text_query)

        grid_size = patch_features.shape[0]
        similarity_map = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                patch_vec = patch_features[i, j]
                # Normalize patch vector
                patch_vec = patch_vec / (np.linalg.norm(patch_vec) + 1e-8)
                # Cosine similarity
                similarity = np.dot(text_features, patch_vec)
                similarity_map[i, j] = similarity

        return similarity_map


class VideoFrameExtractor:
    """
    Extract frames from videos for indexing
    """

    @staticmethod
    def extract_frames(video_path: str, output_dir: str, fps: int = 2,
                      max_frames: Optional[int] = None) -> List[Dict]:
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            fps: Frame extraction rate (frames per second)
            max_frames: Maximum number of frames to extract

        Returns:
            List of frame metadata dicts
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required. Install with: pip install opencv-python")

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)

        frames_metadata = []
        frame_count = 0
        saved_count = 0

        with tqdm(desc=f"Extracting frames from {Path(video_path).name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)

                    timestamp = frame_count / video_fps

                    frames_metadata.append({
                        "frame_id": saved_count,
                        "frame_path": frame_path,
                        "timestamp": timestamp,
                        "frame_number": frame_count
                    })

                    saved_count += 1
                    pbar.update(1)

                    if max_frames and saved_count >= max_frames:
                        break

                frame_count += 1

        cap.release()

        return frames_metadata


class HybridVideoSearch:
    """
    Main class for hybrid video search using CLIP + DINOv3 + LanceDB
    """

    def __init__(self, db_path: str = "video_search.db",
                 clip_model: str = "clip_vit_b32",
                 dino_model: str = "dinov2_vitb14",
                 device: str = "cuda",
                 use_dino: bool = True,
                 use_face_detection: bool = False,
                 face_model: str = "Facenet512",
                 batch_size: int = 32):
        """
        Initialize hybrid video search system with ONNX models

        Args:
            db_path: Path to LanceDB database
            clip_model: CLIP ONNX model variant (clip_vit_b32 or clip_vit_b16)
            dino_model: DINOv3 ONNX model variant (dinov2_vits14 or dinov2_vitb14)
            device: Device to run models on ("cuda" or "cpu")
            use_dino: Whether to use DINOv3 (disable for faster indexing)
            use_face_detection: Whether to enable face detection
            face_model: DeepFace model name (VGG-Face, Facenet, Facenet512, ArcFace, etc.)
            batch_size: Batch size for embedding extraction
        """
        self.db_path = db_path
        self.db = lancedb.connect(db_path)

        # Check ONNX Runtime providers
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        if device == "cuda" and 'CUDAExecutionProvider' not in available_providers:
            print("⚠ CUDA requested but not available in ONNX Runtime, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device

        self.use_dino = use_dino
        self.use_face_detection = use_face_detection
        self.batch_size = batch_size

        # Initialize ONNX CLIP embeddings
        self.clip_embeddings = create_onnx_clip_embeddings(
            model_name=clip_model,
            device=self.device
        )

        # Initialize DINOv3 if enabled
        if use_dino:
            self.dino_embeddings = DINOv3Embeddings(
                model_name=dino_model,
                device=self.device
            )

        # Initialize face detection if enabled
        self.face_embeddings = None
        if use_face_detection:
            if not FACE_DETECTION_AVAILABLE:
                warnings.warn(
                    "Face detection requested but dependencies not available. "
                    "Install with: pip install deepface tf-keras"
                )
            else:
                self.face_embeddings = FaceEmbeddings(
                    model_name=face_model,
                    detector_backend="retinaface",
                    device=self.device
                )
                print("✓ Face detection enabled")

        # Initialize dense feature extractor
        self.dense_extractor = None

        # Frame extractor
        self.frame_extractor = VideoFrameExtractor()

    def index_video(self, video_path: str, video_id: str,
                   fps: int = 2, max_frames: Optional[int] = None,
                   cleanup_frames: bool = False) -> int:
        """
        Index a video by extracting frames and computing embeddings

        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            fps: Frame extraction rate
            max_frames: Maximum number of frames to extract
            cleanup_frames: Whether to delete frames after indexing

        Returns:
            Number of frames indexed
        """
        # Create temp directory for frames
        frames_dir = os.path.join(os.path.dirname(self.db_path), f"frames_{video_id}")

        # Extract frames
        print(f"Extracting frames from {video_path}...")
        frames_metadata = self.frame_extractor.extract_frames(
            video_path, frames_dir, fps=fps, max_frames=max_frames
        )

        if not frames_metadata:
            warnings.warn(f"No frames extracted from {video_path}")
            return 0

        # Prepare data for indexing
        frame_paths = [f["frame_path"] for f in frames_metadata]

        # Extract CLIP embeddings
        print("Extracting CLIP embeddings...")
        clip_embeddings = self.clip_embeddings.compute_source_embeddings(frame_paths)

        # Build CLIP table data
        clip_data = []
        for i, frame_meta in enumerate(frames_metadata):
            clip_data.append({
                "video_id": video_id,
                "frame_id": frame_meta["frame_id"],
                "frame_path": frame_meta["frame_path"],
                "timestamp": frame_meta["timestamp"],
                "frame_number": frame_meta["frame_number"],
                "clip_embedding": clip_embeddings[i]
            })

        # Create or append to CLIP table
        if "clip_embeddings" in self.db.table_names():
            clip_table = self.db.open_table("clip_embeddings")
            clip_table.add(clip_data)
        else:
            self.db.create_table("clip_embeddings", clip_data)

        # Extract DINOv3 embeddings if enabled
        if self.use_dino:
            print("Extracting DINOv3 embeddings...")
            dino_embeddings = self.dino_embeddings.embed_batch(
                frame_paths, batch_size=self.batch_size
            )

            # Build DINOv3 table data
            dino_data = []
            for i, frame_meta in enumerate(frames_metadata):
                dino_data.append({
                    "video_id": video_id,
                    "frame_id": frame_meta["frame_id"],
                    "frame_path": frame_meta["frame_path"],
                    "timestamp": frame_meta["timestamp"],
                    "frame_number": frame_meta["frame_number"],
                    "dino_embedding": dino_embeddings[i]
                })

            # Create or append to DINOv3 table
            if "dino_embeddings" in self.db.table_names():
                dino_table = self.db.open_table("dino_embeddings")
                dino_table.add(dino_data)
            else:
                self.db.create_table("dino_embeddings", dino_data)

        # Cleanup frames if requested
        if cleanup_frames:
            import shutil
            shutil.rmtree(frames_dir)
            print(f"Cleaned up frame directory: {frames_dir}")

        print(f"✓ Indexed {len(frames_metadata)} frames from {video_id}")

        return len(frames_metadata)

    def search_text(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search using text query with CLIP

        Args:
            query: Text description
            limit: Number of results to return

        Returns:
            List of results with frame metadata and scores
        """
        # Generate text embedding
        query_embedding = self.clip_embeddings.generate_text_embeddings(query)

        # Search CLIP table
        clip_table = self.db.open_table("clip_embeddings")
        results = (
            clip_table.search(query_embedding)
            .limit(limit)
            .to_list()
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "video_id": result["video_id"],
                "frame_id": result["frame_id"],
                "frame_path": result["frame_path"],
                "timestamp": result["timestamp"],
                "frame_number": result["frame_number"],
                "score": float(result["_distance"]) if "_distance" in result else 0.0
            })

        return formatted_results

    def search_image(self, image_path: str, limit: int = 10,
                    clip_weight: float = 0.5, dino_weight: float = 0.5) -> List[Dict]:
        """
        Search using image query

        Args:
            image_path: Path to query image
            limit: Number of results
            clip_weight: Weight for CLIP similarity
            dino_weight: Weight for DINOv3 similarity

        Returns:
            List of results
        """
        results_map = {}

        # CLIP search
        clip_embedding = self.clip_embeddings.generate_image_embedding(image_path)
        clip_table = self.db.open_table("clip_embeddings")
        clip_results = (
            clip_table.search(clip_embedding)
            .limit(limit * 2)
            .to_list()
        )

        for result in clip_results:
            key = (result["video_id"], result["frame_id"])
            score = float(result.get("_distance", 0.0))
            results_map[key] = {
                "video_id": result["video_id"],
                "frame_id": result["frame_id"],
                "frame_path": result["frame_path"],
                "timestamp": result["timestamp"],
                "frame_number": result["frame_number"],
                "score": clip_weight * score
            }

        # DINOv3 search if enabled
        if self.use_dino:
            dino_embedding = self.dino_embeddings.embed_image(image_path)
            dino_table = self.db.open_table("dino_embeddings")
            dino_results = (
                dino_table.search(dino_embedding)
                .limit(limit * 2)
                .to_list()
            )

            for result in dino_results:
                key = (result["video_id"], result["frame_id"])
                score = float(result.get("_distance", 0.0))

                if key in results_map:
                    results_map[key]["score"] += dino_weight * score
                else:
                    results_map[key] = {
                        "video_id": result["video_id"],
                        "frame_id": result["frame_id"],
                        "frame_path": result["frame_path"],
                        "timestamp": result["timestamp"],
                        "frame_number": result["frame_number"],
                        "score": dino_weight * score
                    }

        # Sort by combined score and return top results
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return sorted_results[:limit]

    def hybrid_search(self, text: Optional[str] = None,
                     image: Optional[str] = None,
                     text_weight: float = 0.5,
                     image_weight: float = 0.5,
                     limit: int = 10) -> List[Dict]:
        """
        Combined text and image search

        Args:
            text: Text query (optional)
            image: Image path (optional)
            text_weight: Weight for text results
            image_weight: Weight for image results
            limit: Number of results

        Returns:
            List of combined results
        """
        if not text and not image:
            raise ValueError("At least one of text or image must be provided")

        results_map = {}

        # Text search
        if text:
            text_results = self.search_text(text, limit=limit * 2)
            for result in text_results:
                key = (result["video_id"], result["frame_id"])
                results_map[key] = result.copy()
                results_map[key]["score"] = text_weight * result["score"]

        # Image search
        if image:
            image_results = self.search_image(image, limit=limit * 2)
            for result in image_results:
                key = (result["video_id"], result["frame_id"])
                if key in results_map:
                    results_map[key]["score"] += image_weight * result["score"]
                else:
                    results_map[key] = result.copy()
                    results_map[key]["score"] = image_weight * result["score"]

        # Sort and return top results
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return sorted_results[:limit]

    def locate_object(self, frame_path: str, query: str,
                     threshold: float = 0.25) -> List[Dict]:
        """
        Find and localize objects in a frame using text query

        Args:
            frame_path: Path to frame image
            query: Text description of object
            threshold: Detection confidence threshold

        Returns:
            List of detections with bounding boxes and scores
        """
        # Initialize dense extractor if needed
        if self.dense_extractor is None:
            self.dense_extractor = DenseFeatureExtractor(device=self.device)

        # Extract dense features
        patch_features, grid_size = self.dense_extractor.extract_dense_clip(frame_path)

        # Compute similarity map
        similarity_map = self.dense_extractor.compute_similarity_map(
            patch_features, query
        )

        # Find regions above threshold
        detections = []

        # Try different window sizes for multi-scale detection
        img = Image.open(frame_path)
        img_width, img_height = img.size

        for win_h in [2, 3, 4]:
            for win_w in [2, 3, 4]:
                for i in range(grid_size - win_h + 1):
                    for j in range(grid_size - win_w + 1):
                        # Compute average similarity in window
                        window_sim = similarity_map[i:i+win_h, j:j+win_w].mean()

                        if window_sim > threshold:
                            # Convert to pixel coordinates
                            patch_size = img_width / grid_size
                            x = int(j * patch_size)
                            y = int(i * patch_size)
                            w = int(win_w * patch_size)
                            h = int(win_h * patch_size)

                            detections.append({
                                "bbox": (x, y, w, h),
                                "score": float(window_sim),
                                "grid_coords": (i, j, win_h, win_w)
                            })

        # Apply Non-Maximum Suppression
        detections = self._nms(detections, iou_threshold=0.5)

        return detections

    @staticmethod
    def _nms(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections

        Args:
            detections: List of detection dicts with bbox and score
            iou_threshold: IoU threshold for suppression

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        # Sort by score
        detections = sorted(detections, key=lambda x: x["score"], reverse=True)

        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            detections = [
                det for det in detections
                if HybridVideoSearch._compute_iou(best["bbox"], det["bbox"]) < iou_threshold
            ]

        return keep

    @staticmethod
    def _compute_iou(bbox1: Tuple[int, int, int, int],
                    bbox2: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union between two bounding boxes

        Args:
            bbox1: (x, y, w, h)
            bbox2: (x, y, w, h)

        Returns:
            IoU value
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Compute intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def visualize_detections(self, frame_path: str, detections: List[Dict],
                           output_path: Optional[str] = None) -> Image.Image:
        """
        Visualize detections on frame

        Args:
            frame_path: Path to frame image
            detections: List of detections from locate_object
            output_path: Optional path to save visualization

        Returns:
            PIL Image with visualized detections
        """
        img = Image.open(frame_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        for det in detections:
            x, y, w, h = det["bbox"]
            score = det["score"]

            # Draw bounding box
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

            # Draw score
            text = f"{score:.2f}"
            draw.text((x, y - 20), text, fill="red", font=font)

        if output_path:
            img.save(output_path)
            print(f"Saved visualization to {output_path}")

        return img

    def get_stats(self) -> Dict:
        """
        Get database statistics

        Returns:
            Dict with statistics
        """
        stats = {
            "tables": self.db.table_names(),
            "total_frames": 0,
            "videos": set(),
            "total_faces": 0
        }

        if "clip_embeddings" in self.db.table_names():
            clip_table = self.db.open_table("clip_embeddings")
            df = clip_table.to_pandas()
            stats["total_frames"] = len(df)
            stats["videos"] = set(df["video_id"].unique())

        if "face_embeddings" in self.db.table_names():
            face_table = self.db.open_table("face_embeddings")
            df_faces = face_table.to_pandas()
            stats["total_faces"] = len(df_faces)

        return stats

    # ==================== Face Detection Methods ====================

    def index_video_faces(self, video_path: str, video_id: str,
                         fps: int = 2, max_frames: Optional[int] = None,
                         min_confidence: float = 0.5,
                         frames_dir: Optional[str] = None) -> int:
        """
        Index faces from a video

        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            fps: Frame extraction rate
            max_frames: Maximum number of frames to extract
            min_confidence: Minimum face detection confidence
            frames_dir: Directory containing pre-extracted frames (optional)

        Returns:
            Number of faces indexed
        """
        if not self.face_embeddings:
            raise ValueError("Face detection not enabled. Initialize with use_face_detection=True")

        # Extract frames if not provided
        if frames_dir is None:
            frames_dir = os.path.join(os.path.dirname(self.db_path), f"frames_{video_id}")
            print(f"Extracting frames from {video_path}...")
            frames_metadata = self.frame_extractor.extract_frames(
                video_path, frames_dir, fps=fps, max_frames=max_frames
            )
        else:
            # Load existing frames
            frame_files = sorted(Path(frames_dir).glob("*.jpg"))
            frames_metadata = []
            for i, frame_file in enumerate(frame_files):
                frames_metadata.append({
                    "frame_id": i,
                    "frame_path": str(frame_file),
                    "timestamp": i / fps,
                    "frame_number": i
                })

        if not frames_metadata:
            warnings.warn(f"No frames found for face detection")
            return 0

        # Detect faces in all frames
        print("Detecting faces in frames...")
        face_data = []
        total_faces = 0

        for frame_meta in tqdm(frames_metadata, desc="Processing frames"):
            frame_path = frame_meta["frame_path"]

            # Detect faces
            detections = self.face_embeddings.detect_faces(frame_path, min_confidence)

            for face_idx, det in enumerate(detections):
                face_data.append({
                    "video_id": video_id,
                    "frame_id": frame_meta["frame_id"],
                    "frame_path": frame_path,
                    "timestamp": frame_meta["timestamp"],
                    "frame_number": frame_meta["frame_number"],
                    "face_idx": face_idx,
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "face_embedding": det["embedding"],
                    "age": det.get("age"),
                    "gender": det.get("gender")
                })
                total_faces += 1

        if not face_data:
            print("No faces detected in video")
            return 0

        # Create or append to face embeddings table
        if "face_embeddings" in self.db.table_names():
            face_table = self.db.open_table("face_embeddings")
            face_table.add(face_data)
        else:
            self.db.create_table("face_embeddings", face_data)

        print(f"✓ Indexed {total_faces} faces from {len(frames_metadata)} frames")

        return total_faces

    def search_faces(self, query_face: Union[str, np.ndarray],
                    min_confidence: float = 0.5,
                    similarity_threshold: float = 0.6,
                    limit: int = 10) -> List[Dict]:
        """
        Search for similar faces in the database

        Args:
            query_face: Image path or face embedding
            min_confidence: Minimum face detection confidence (if image)
            similarity_threshold: Minimum similarity score
            limit: Number of results to return

        Returns:
            List of matching faces with metadata
        """
        if not self.face_embeddings:
            raise ValueError("Face detection not enabled")

        if "face_embeddings" not in self.db.table_names():
            return []

        # Extract query embedding if image path provided
        if isinstance(query_face, str):
            query_embedding = self.face_embeddings.extract_largest_face_embedding(
                query_face, min_confidence
            )
            if query_embedding is None:
                warnings.warn("No face detected in query image")
                return []
        else:
            query_embedding = query_face

        # Search face table
        face_table = self.db.open_table("face_embeddings")
        results = (
            face_table.search(query_embedding)
            .limit(limit * 2)  # Get more to filter by threshold
            .to_list()
        )

        # Filter by similarity threshold and format results
        formatted_results = []
        for result in results:
            similarity = float(result.get("_distance", 0.0))

            if similarity >= similarity_threshold:
                formatted_results.append({
                    "video_id": result["video_id"],
                    "frame_id": result["frame_id"],
                    "frame_path": result["frame_path"],
                    "timestamp": result["timestamp"],
                    "frame_number": result["frame_number"],
                    "face_idx": result["face_idx"],
                    "bbox": result["bbox"],
                    "confidence": result["confidence"],
                    "similarity": similarity,
                    "age": result.get("age"),
                    "gender": result.get("gender")
                })

                if len(formatted_results) >= limit:
                    break

        return formatted_results

    def find_person_across_video(self, reference_face: Union[str, np.ndarray],
                                video_id: Optional[str] = None,
                                similarity_threshold: float = 0.6,
                                limit: int = 100) -> List[Dict]:
        """
        Track a person across video frames

        Args:
            reference_face: Reference image or face embedding
            video_id: Specific video to search (or None for all)
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results

        Returns:
            List of appearances sorted by timestamp
        """
        # Search for similar faces
        results = self.search_faces(
            reference_face,
            similarity_threshold=similarity_threshold,
            limit=limit
        )

        # Filter by video if specified
        if video_id:
            results = [r for r in results if r["video_id"] == video_id]

        # Sort by timestamp
        results = sorted(results, key=lambda x: x["timestamp"])

        return results

    def detect_faces_in_frame(self, frame_path: str,
                             min_confidence: float = 0.5,
                             visualize: bool = False,
                             output_path: Optional[str] = None) -> List[Dict]:
        """
        Detect faces in a single frame

        Args:
            frame_path: Path to frame image
            min_confidence: Minimum detection confidence
            visualize: Whether to create visualization
            output_path: Path to save visualization

        Returns:
            List of face detections
        """
        if not self.face_embeddings:
            raise ValueError("Face detection not enabled")

        detections = self.face_embeddings.detect_faces(frame_path, min_confidence)

        if visualize:
            self.face_embeddings.visualize_faces(
                frame_path,
                detections,
                output_path=output_path
            )

        return detections

    def extract_face_crops(self, frame_path: str,
                          min_confidence: float = 0.5,
                          output_dir: Optional[str] = None) -> List[Tuple[Image.Image, Dict]]:
        """
        Extract and optionally save face crops from a frame

        Args:
            frame_path: Path to frame image
            min_confidence: Minimum detection confidence
            output_dir: Optional directory to save crops

        Returns:
            List of (face_image, detection_info) tuples
        """
        if not self.face_embeddings:
            raise ValueError("Face detection not enabled")

        crops = self.face_embeddings.extract_face_crops(frame_path, min_confidence)

        # Save crops if output directory specified
        if output_dir and crops:
            os.makedirs(output_dir, exist_ok=True)
            frame_name = Path(frame_path).stem

            for i, (crop_img, det) in enumerate(crops):
                crop_path = os.path.join(output_dir, f"{frame_name}_face_{i}.jpg")
                crop_img.save(crop_path)

            print(f"✓ Saved {len(crops)} face crops to {output_dir}")

        return crops

    def get_face_statistics(self, video_id: Optional[str] = None) -> Dict:
        """
        Get face detection statistics

        Args:
            video_id: Optional video ID to filter by

        Returns:
            Dictionary with statistics
        """
        if "face_embeddings" not in self.db.table_names():
            return {
                "total_faces": 0,
                "unique_frames": 0,
                "avg_faces_per_frame": 0
            }

        face_table = self.db.open_table("face_embeddings")
        df = face_table.to_pandas()

        if video_id:
            df = df[df["video_id"] == video_id]

        stats = {
            "total_faces": len(df),
            "unique_frames": df["frame_id"].nunique(),
            "avg_faces_per_frame": len(df) / df["frame_id"].nunique() if len(df) > 0 else 0,
            "videos": list(df["video_id"].unique()) if not video_id else [video_id]
        }

        # Add gender distribution if available
        if "gender" in df.columns:
            gender_counts = df["gender"].value_counts().to_dict()
            stats["gender_distribution"] = gender_counts

        # Add age statistics if available
        if "age" in df.columns:
            stats["age_min"] = int(df["age"].min())
            stats["age_max"] = int(df["age"].max())
            stats["age_mean"] = float(df["age"].mean())

        return stats


def main():
    """
    Example usage
    """
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Video Search")
    parser.add_argument("--index", type=str, help="Path to video to index")
    parser.add_argument("--video-id", type=str, help="Video ID")
    parser.add_argument("--search-text", type=str, help="Text query to search")
    parser.add_argument("--search-image", type=str, help="Image path to search")
    parser.add_argument("--db", type=str, default="video_search.db", help="Database path")
    parser.add_argument("--fps", type=int, default=2, help="Frame extraction rate")
    parser.add_argument("--limit", type=int, default=10, help="Number of results")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Initialize search system
    search = HybridVideoSearch(db_path=args.db, device=args.device)

    # Index video if provided
    if args.index and args.video_id:
        search.index_video(args.index, args.video_id, fps=args.fps)

    # Search with text
    if args.search_text:
        results = search.search_text(args.search_text, limit=args.limit)
        print(f"\nText search results for '{args.search_text}':")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['video_id']} - Frame {result['frame_id']} "
                  f"at {result['timestamp']:.2f}s (score: {result['score']:.3f})")

    # Search with image
    if args.search_image:
        results = search.search_image(args.search_image, limit=args.limit)
        print(f"\nImage search results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['video_id']} - Frame {result['frame_id']} "
                  f"at {result['timestamp']:.2f}s (score: {result['score']:.3f})")

    # Show stats
    stats = search.get_stats()
    print(f"\nDatabase stats:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Videos: {len(stats['videos'])}")


if __name__ == "__main__":
    main()
