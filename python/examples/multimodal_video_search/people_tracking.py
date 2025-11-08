"""
People Tracking using YOLO ONNX + Roboflow Supervision

This module provides person detection and tracking capabilities using:
- YOLOv8 (ONNX) for person detection
- Roboflow Supervision for multi-object tracking (ByteTrack)
- Face embeddings for person identification
- LanceDB for storing track information

Features:
- Detect and track people across video frames
- Assign unique IDs to each person
- Extract appearance embeddings for re-identification
- Track people across multiple videos
- Search for specific individuals
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX Runtime not available. Install with: pip install onnxruntime-gpu")

try:
    import supervision as sv
    from supervision.tracker.byte_tracker.core import ByteTrack
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    warnings.warn("Supervision not available. Install with: pip install supervision")

# Optional face embeddings for person identification
try:
    from face_embeddings import FaceEmbeddings
    FACE_AVAILABLE = True
except ImportError:
    FACE_AVAILABLE = False
    FaceEmbeddings = None


@dataclass
class PersonTrack:
    """Represents a tracked person across frames"""
    track_id: int
    video_id: str
    first_frame: int
    last_frame: int
    first_timestamp: float
    last_timestamp: float
    bboxes: List[List[float]]  # List of [x, y, w, h] for each frame
    frame_ids: List[int]
    timestamps: List[float]
    confidences: List[float]
    appearance_embedding: Optional[np.ndarray] = None
    face_embedding: Optional[np.ndarray] = None
    person_id: Optional[str] = None  # For identified persons

    def duration(self) -> float:
        """Get the duration this person appears in the video"""
        return self.last_timestamp - self.first_timestamp

    def num_detections(self) -> int:
        """Get the number of frames this person appears in"""
        return len(self.frame_ids)


class YOLOv8ONNX:
    """
    YOLOv8 ONNX inference for person detection

    Uses pre-exported YOLOv8 ONNX models for efficient inference
    """

    def __init__(self, model_path: Optional[str] = None,
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 providers: Optional[List[str]] = None):
        """
        Initialize YOLOv8 ONNX model

        Args:
            model_path: Path to ONNX model file (if None, will download YOLOv8n)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            providers: ONNX Runtime execution providers
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime required. Install with: pip install onnxruntime-gpu")

        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Download or use provided model
        if model_path is None:
            model_path = self._download_yolov8_onnx()

        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2]  # Usually 640

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # COCO person class ID is 0
        self.person_class_id = 0

    def _download_yolov8_onnx(self) -> str:
        """Download YOLOv8n ONNX model"""
        from huggingface_hub import hf_hub_download

        cache_dir = Path.home() / ".cache" / "yolo_onnx"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_file = cache_dir / "yolov8n.onnx"

        if not model_file.exists():
            print("Downloading YOLOv8n ONNX model from HuggingFace...")
            try:
                # Download from HuggingFace
                downloaded = hf_hub_download(
                    repo_id="arnabdhar/YOLOv8-ONNX-models",
                    filename="yolov8n.onnx",
                    cache_dir=cache_dir
                )
                # Move to expected location
                import shutil
                shutil.copy(downloaded, model_file)
                print(f"✓ Model downloaded to {model_file}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Please download YOLOv8 ONNX model manually and provide path")
                raise

        return str(model_file)

    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for YOLO inference

        Args:
            image: Input image (PIL or numpy array)

        Returns:
            Preprocessed image tensor, scale factor, original size
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Get original size
        orig_h, orig_w = image.shape[:2]

        # Resize to model input size (letterbox)
        img_resized = cv2.resize(image, (self.img_size, self.img_size))

        # Convert BGR to RGB if needed
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 4:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGB)

        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Transpose to CHW format
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension
        img_input = np.expand_dims(img_transposed, axis=0)

        scale = self.img_size / max(orig_w, orig_h)

        return img_input, scale, (orig_w, orig_h)

    def postprocess(self, outputs: np.ndarray, scale: float,
                   orig_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process YOLO outputs

        Args:
            outputs: Model outputs
            scale: Scale factor from preprocessing
            orig_size: Original image size (w, h)

        Returns:
            boxes (N, 4), scores (N,), class_ids (N,)
        """
        # YOLOv8 output shape: [1, 84, 8400] -> [8400, 84]
        predictions = outputs[0].transpose(1, 0)

        # Extract boxes and scores
        boxes = predictions[:, :4]
        scores = predictions[:, 4:].max(axis=1)
        class_ids = predictions[:, 4:].argmax(axis=1)

        # Filter by confidence
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Filter only person class
        person_mask = class_ids == self.person_class_id
        boxes = boxes[person_mask]
        scores = scores[person_mask]
        class_ids = class_ids[person_mask]

        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Convert from xywh to xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Scale back to original size
        boxes_xyxy = boxes_xyxy / scale

        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_size[0])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_size[1])

        # Apply NMS
        indices = self._nms(boxes_xyxy, scores, self.iou_threshold)

        return boxes_xyxy[indices], scores[indices], class_ids[indices]

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect persons in image

        Args:
            image: Input image

        Returns:
            boxes (N, 4) in xyxy format, confidences (N,)
        """
        # Preprocess
        img_input, scale, orig_size = self.preprocess(image)

        # Run inference
        outputs = self.session.run(None, {self.input_name: img_input})

        # Postprocess
        boxes, scores, _ = self.postprocess(outputs[0], scale, orig_size)

        return boxes, scores


class PeopleTracker:
    """
    Multi-person tracking using YOLO + Roboflow Supervision

    Tracks people across video frames and maintains consistent IDs
    """

    def __init__(self,
                 yolo_model_path: Optional[str] = None,
                 conf_threshold: float = 0.5,
                 device: str = "cuda",
                 use_face_id: bool = False,
                 face_model: str = "Facenet512"):
        """
        Initialize people tracker

        Args:
            yolo_model_path: Path to YOLO ONNX model
            conf_threshold: Confidence threshold for detections
            device: Device for inference ("cuda" or "cpu")
            use_face_id: Whether to use face recognition for person ID
            face_model: Face recognition model name
        """
        if not SUPERVISION_AVAILABLE:
            raise ImportError("Supervision required. Install with: pip install supervision")

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

        # Initialize YOLO detector
        self.detector = YOLOv8ONNX(
            model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            providers=providers
        )

        # Initialize ByteTrack tracker
        self.tracker = ByteTrack()

        # Track storage
        self.tracks: Dict[int, PersonTrack] = {}
        self.track_counter = 0

        # Optional face recognition
        self.use_face_id = use_face_id
        self.face_extractor = None
        if use_face_id and FACE_AVAILABLE:
            self.face_extractor = FaceEmbeddings(
                model_name=face_model,
                detector_backend="retinaface",
                device=device
            )
            print("✓ Face recognition enabled for person identification")
        elif use_face_id:
            warnings.warn("Face recognition requested but not available")

    def track_frame(self,
                    frame: Union[np.ndarray, Image.Image],
                    frame_id: int,
                    timestamp: float,
                    video_id: str) -> sv.Detections:
        """
        Track people in a single frame

        Args:
            frame: Input frame
            frame_id: Frame number
            timestamp: Timestamp in video
            video_id: Video identifier

        Returns:
            Supervision Detections with tracking IDs
        """
        # Convert PIL to numpy if needed
        if isinstance(frame, Image.Image):
            frame = np.array(frame)

        # Detect people
        boxes, confidences = self.detector.detect(frame)

        if len(boxes) == 0:
            # Return empty detections
            return sv.Detections.empty()

        # Create Supervision Detections
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=np.zeros(len(boxes), dtype=int)  # All are person class
        )

        # Update tracker
        detections = self.tracker.update_with_detections(detections)

        # Update track history
        if detections.tracker_id is not None:
            for i, track_id in enumerate(detections.tracker_id):
                bbox = boxes[i]
                conf = confidences[i]

                # Convert xyxy to xywh
                x1, y1, x2, y2 = bbox
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                if track_id not in self.tracks:
                    # New track
                    self.tracks[track_id] = PersonTrack(
                        track_id=int(track_id),
                        video_id=video_id,
                        first_frame=frame_id,
                        last_frame=frame_id,
                        first_timestamp=timestamp,
                        last_timestamp=timestamp,
                        bboxes=[[float(x), float(y), float(w), float(h)]],
                        frame_ids=[frame_id],
                        timestamps=[timestamp],
                        confidences=[float(conf)]
                    )
                else:
                    # Update existing track
                    track = self.tracks[track_id]
                    track.last_frame = frame_id
                    track.last_timestamp = timestamp
                    track.bboxes.append([float(x), float(y), float(w), float(h)])
                    track.frame_ids.append(frame_id)
                    track.timestamps.append(timestamp)
                    track.confidences.append(float(conf))

        return detections

    def track_video(self,
                    video_path: str,
                    video_id: str,
                    fps: int = 2,
                    max_frames: Optional[int] = None) -> List[PersonTrack]:
        """
        Track people throughout an entire video

        Args:
            video_path: Path to video file
            video_id: Video identifier
            fps: Frame extraction rate
            max_frames: Maximum frames to process

        Returns:
            List of PersonTrack objects
        """
        # Reset tracker for new video
        self.tracker = ByteTrack()
        self.tracks = {}

        # Open video
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)

        frame_count = 0
        processed_count = 0

        print(f"Tracking people in {Path(video_path).name}...")

        with tqdm(desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    timestamp = frame_count / video_fps

                    # Track people in this frame
                    self.track_frame(frame, processed_count, timestamp, video_id)

                    processed_count += 1
                    pbar.update(1)

                    if max_frames and processed_count >= max_frames:
                        break

                frame_count += 1

        cap.release()

        # Return all tracks
        tracks = list(self.tracks.values())
        print(f"✓ Found {len(tracks)} unique people")

        return tracks

    def get_track_statistics(self) -> Dict:
        """Get statistics about current tracks"""
        if not self.tracks:
            return {
                'num_tracks': 0,
                'avg_duration': 0,
                'avg_detections': 0
            }

        durations = [t.duration() for t in self.tracks.values()]
        detections = [t.num_detections() for t in self.tracks.values()]

        return {
            'num_tracks': len(self.tracks),
            'avg_duration': np.mean(durations),
            'max_duration': np.max(durations),
            'avg_detections': np.mean(detections),
            'max_detections': np.max(detections)
        }

    def visualize_tracks(self,
                        frame: np.ndarray,
                        detections: sv.Detections,
                        show_labels: bool = True) -> np.ndarray:
        """
        Visualize tracking results on a frame

        Args:
            frame: Input frame
            detections: Detections with tracking IDs
            show_labels: Whether to show track ID labels

        Returns:
            Frame with visualizations
        """
        # Create annotators
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Annotate frame
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        if show_labels and detections.tracker_id is not None:
            labels = [f"ID: {tid}" for tid in detections.tracker_id]
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

        return annotated_frame

    def export_tracks(self, output_path: str):
        """Export tracks to JSON file"""
        import json

        tracks_data = {
            str(tid): {
                'track_id': track.track_id,
                'video_id': track.video_id,
                'first_frame': track.first_frame,
                'last_frame': track.last_frame,
                'first_timestamp': track.first_timestamp,
                'last_timestamp': track.last_timestamp,
                'duration': track.duration(),
                'num_detections': track.num_detections(),
                'bboxes': track.bboxes,
                'frame_ids': track.frame_ids,
                'timestamps': track.timestamps,
                'confidences': track.confidences
            }
            for tid, track in self.tracks.items()
        }

        with open(output_path, 'w') as f:
            json.dump(tracks_data, f, indent=2)

        print(f"✓ Exported {len(tracks_data)} tracks to {output_path}")
