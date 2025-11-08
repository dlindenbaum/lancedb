"""
Person Clustering and Re-Identification

This module combines people tracking with face recognition to:
- Extract faces from tracked people
- Cluster faces to identify the same person
- Re-identify people across tracks and videos
- Link tracks belonging to the same person
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠ scikit-learn not available. Install with: pip install scikit-learn")

try:
    from face_embeddings import FaceEmbeddings
    FACE_AVAILABLE = True
except ImportError:
    FACE_AVAILABLE = False
    FaceEmbeddings = None

try:
    from people_tracking import PersonTrack
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False
    PersonTrack = None


@dataclass
class PersonCluster:
    """Represents a cluster of tracks belonging to the same person"""
    cluster_id: int
    person_name: Optional[str] = None
    track_ids: List[int] = None
    video_ids: List[str] = None
    face_embeddings: List[np.ndarray] = None
    representative_embedding: Optional[np.ndarray] = None
    confidence: float = 0.0

    def __post_init__(self):
        if self.track_ids is None:
            self.track_ids = []
        if self.video_ids is None:
            self.video_ids = []
        if self.face_embeddings is None:
            self.face_embeddings = []

    def add_track(self, track_id: int, video_id: str, face_embedding: np.ndarray):
        """Add a track to this cluster"""
        self.track_ids.append(track_id)
        self.video_ids.append(video_id)
        self.face_embeddings.append(face_embedding)

        # Update representative embedding (mean of all embeddings)
        if len(self.face_embeddings) > 0:
            self.representative_embedding = np.mean(self.face_embeddings, axis=0)

    def num_tracks(self) -> int:
        """Number of tracks in this cluster"""
        return len(self.track_ids)

    def num_videos(self) -> int:
        """Number of unique videos this person appears in"""
        return len(set(self.video_ids))


class PersonClusterer:
    """
    Cluster tracked people using face recognition

    This class extracts faces from tracked person bounding boxes,
    computes face embeddings, and clusters them to identify unique individuals.
    """

    def __init__(self,
                 face_model: str = "Facenet512",
                 detector_backend: str = "retinaface",
                 clustering_method: str = "dbscan",
                 similarity_threshold: float = 0.6,
                 min_face_confidence: float = 0.5,
                 device: str = "cuda"):
        """
        Initialize person clusterer

        Args:
            face_model: Face recognition model (Facenet512, ArcFace, etc.)
            detector_backend: Face detector (retinaface, mtcnn, opencv, etc.)
            clustering_method: Clustering algorithm (dbscan or agglomerative)
            similarity_threshold: Minimum face similarity for same person
            min_face_confidence: Minimum face detection confidence
            device: Device for face detection
        """
        if not FACE_AVAILABLE:
            raise ImportError("Face embeddings required. Ensure face_embeddings.py is available")

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        self.face_extractor = FaceEmbeddings(
            model_name=face_model,
            detector_backend=detector_backend,
            device=device
        )

        self.clustering_method = clustering_method
        self.similarity_threshold = similarity_threshold
        self.min_face_confidence = min_face_confidence

        # Storage
        self.clusters: Dict[int, PersonCluster] = {}
        self.track_to_cluster: Dict[Tuple[int, str], int] = {}  # (track_id, video_id) -> cluster_id

    def extract_face_from_track(self,
                                video_path: str,
                                track: PersonTrack,
                                max_faces: int = 5) -> Optional[np.ndarray]:
        """
        Extract face embedding from a tracked person

        Samples frames from the track and extracts the best face embedding

        Args:
            video_path: Path to video file
            track: PersonTrack object
            max_faces: Maximum number of frames to sample

        Returns:
            Face embedding or None if no face detected
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # Sample frames evenly from the track
        num_frames = len(track.frame_ids)
        step = max(1, num_frames // max_faces)
        sample_indices = list(range(0, num_frames, step))[:max_faces]

        face_embeddings = []
        face_confidences = []

        for idx in sample_indices:
            timestamp = track.timestamps[idx]
            bbox = track.bboxes[idx]

            # Seek to frame
            frame_num = int(timestamp * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                continue

            # Crop to person bbox (with some padding)
            x, y, w, h = bbox
            padding = 0.1  # 10% padding
            x1 = max(0, int(x - w * padding))
            y1 = max(0, int(y - h * padding))
            x2 = min(frame.shape[1], int(x + w * (1 + padding)))
            y2 = min(frame.shape[0], int(y + h * (1 + padding)))

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            # Convert to PIL Image
            person_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))

            # Detect faces in the crop
            try:
                detections = self.face_extractor.detect_faces_from_image(
                    person_img,
                    min_confidence=self.min_face_confidence
                )

                if detections:
                    # Use the largest face (highest confidence)
                    best_det = max(detections, key=lambda d: d['confidence'])
                    face_embeddings.append(best_det['embedding'])
                    face_confidences.append(best_det['confidence'])
            except Exception as e:
                # Face detection might fail, continue
                continue

        cap.release()

        if not face_embeddings:
            return None

        # Return the embedding with highest confidence
        best_idx = np.argmax(face_confidences)
        return face_embeddings[best_idx]

    def cluster_tracks(self,
                      tracks: List[PersonTrack],
                      video_paths: Dict[str, str],
                      extract_faces: bool = True) -> Dict[int, PersonCluster]:
        """
        Cluster tracks to identify unique individuals

        Args:
            tracks: List of PersonTrack objects
            video_paths: Dict mapping video_id to video file path
            extract_faces: Whether to extract faces (or use existing embeddings)

        Returns:
            Dictionary of cluster_id -> PersonCluster
        """
        print(f"Clustering {len(tracks)} tracks...")

        # Extract face embeddings for each track
        track_embeddings = []
        track_info = []  # (track_id, video_id)

        for track in tqdm(tracks, desc="Extracting faces"):
            # Check if track already has face embedding
            if track.face_embedding is not None:
                embedding = track.face_embedding
            elif extract_faces:
                video_path = video_paths.get(track.video_id)
                if video_path is None:
                    print(f"⚠ Video path not found for {track.video_id}, skipping track {track.track_id}")
                    continue

                embedding = self.extract_face_from_track(video_path, track)
                if embedding is None:
                    print(f"⚠ No face found in track {track.track_id}, skipping")
                    continue
            else:
                print(f"⚠ Track {track.track_id} has no face embedding, skipping")
                continue

            track_embeddings.append(embedding)
            track_info.append((track.track_id, track.video_id))

        if len(track_embeddings) == 0:
            print("No face embeddings extracted, cannot cluster")
            return {}

        # Stack embeddings
        embeddings_matrix = np.vstack(track_embeddings)

        # Perform clustering
        print(f"Clustering {len(track_embeddings)} face embeddings...")

        if self.clustering_method == "dbscan":
            # DBSCAN clustering
            # Convert similarity threshold to distance threshold
            distance_threshold = 1.0 - self.similarity_threshold

            clusterer = DBSCAN(
                eps=distance_threshold,
                min_samples=1,  # Allow single-sample clusters
                metric='cosine'
            )
            labels = clusterer.fit_predict(embeddings_matrix)

        elif self.clustering_method == "agglomerative":
            # Agglomerative clustering
            distance_threshold = 1.0 - self.similarity_threshold

            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='cosine',
                linkage='average'
            )
            labels = clusterer.fit_predict(embeddings_matrix)

        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        # Build clusters
        clusters = {}
        track_to_cluster = {}

        for idx, cluster_id in enumerate(labels):
            cluster_id = int(cluster_id)
            track_id, video_id = track_info[idx]
            embedding = track_embeddings[idx]

            # Create cluster if doesn't exist
            if cluster_id not in clusters:
                clusters[cluster_id] = PersonCluster(cluster_id=cluster_id)

            # Add track to cluster
            clusters[cluster_id].add_track(track_id, video_id, embedding)
            track_to_cluster[(track_id, video_id)] = cluster_id

        # Calculate cluster confidences (average pairwise similarity)
        for cluster in clusters.values():
            if len(cluster.face_embeddings) > 1:
                # Compute pairwise similarities
                sims = cosine_similarity(cluster.face_embeddings)
                # Average of upper triangle (exclude diagonal)
                cluster.confidence = float(np.mean(sims[np.triu_indices_from(sims, k=1)]))
            else:
                cluster.confidence = 1.0  # Single track = perfect confidence

        print(f"✓ Found {len(clusters)} unique people")

        self.clusters = clusters
        self.track_to_cluster = track_to_cluster

        return clusters

    def identify_person(self,
                       query_image: str,
                       min_similarity: float = 0.6) -> Optional[Tuple[int, float]]:
        """
        Identify which cluster a person belongs to

        Args:
            query_image: Path to image of person
            min_similarity: Minimum similarity threshold

        Returns:
            (cluster_id, similarity) or None if no match
        """
        # Extract face from query image
        query_embedding = self.face_extractor.extract_largest_face_embedding(
            query_image,
            min_confidence=self.min_face_confidence
        )

        if query_embedding is None:
            print("No face detected in query image")
            return None

        # Compare with all cluster representatives
        best_cluster = None
        best_similarity = -1.0

        for cluster_id, cluster in self.clusters.items():
            if cluster.representative_embedding is None:
                continue

            # Compute cosine similarity
            similarity = float(cosine_similarity(
                [query_embedding],
                [cluster.representative_embedding]
            )[0, 0])

            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id

        if best_similarity >= min_similarity:
            return (best_cluster, best_similarity)
        else:
            return None

    def assign_person_names(self, name_mapping: Dict[int, str]):
        """
        Assign names to clusters

        Args:
            name_mapping: Dict mapping cluster_id -> person_name
        """
        for cluster_id, name in name_mapping.items():
            if cluster_id in self.clusters:
                self.clusters[cluster_id].person_name = name

    def get_cluster_statistics(self) -> Dict:
        """Get statistics about clusters"""
        if not self.clusters:
            return {
                'num_clusters': 0,
                'avg_tracks_per_person': 0,
                'avg_videos_per_person': 0
            }

        num_tracks = [c.num_tracks() for c in self.clusters.values()]
        num_videos = [c.num_videos() for c in self.clusters.values()]

        return {
            'num_clusters': len(self.clusters),
            'total_tracks': sum(num_tracks),
            'avg_tracks_per_person': np.mean(num_tracks),
            'max_tracks_per_person': max(num_tracks),
            'avg_videos_per_person': np.mean(num_videos),
            'max_videos_per_person': max(num_videos),
            'avg_cluster_confidence': np.mean([c.confidence for c in self.clusters.values()])
        }

    def export_clusters(self, output_path: str):
        """Export cluster information to JSON"""
        import json

        clusters_data = {}
        for cluster_id, cluster in self.clusters.items():
            clusters_data[str(cluster_id)] = {
                'cluster_id': cluster.cluster_id,
                'person_name': cluster.person_name,
                'num_tracks': cluster.num_tracks(),
                'num_videos': cluster.num_videos(),
                'track_ids': cluster.track_ids,
                'video_ids': cluster.video_ids,
                'confidence': cluster.confidence
            }

        with open(output_path, 'w') as f:
            json.dump(clusters_data, f, indent=2)

        print(f"✓ Exported {len(clusters_data)} clusters to {output_path}")

    def visualize_cluster(self,
                         cluster_id: int,
                         video_paths: Dict[str, str],
                         tracks_dict: Dict[Tuple[int, str], PersonTrack],
                         output_path: str,
                         max_images: int = 10):
        """
        Create a visualization showing all tracks in a cluster

        Args:
            cluster_id: Cluster ID to visualize
            video_paths: Dict mapping video_id -> video_path
            tracks_dict: Dict mapping (track_id, video_id) -> PersonTrack
            output_path: Where to save visualization
            max_images: Maximum number of images to show
        """
        if cluster_id not in self.clusters:
            print(f"Cluster {cluster_id} not found")
            return

        cluster = self.clusters[cluster_id]

        # Collect representative frames from each track
        images = []
        labels = []

        for track_id, video_id in zip(cluster.track_ids, cluster.video_ids):
            if len(images) >= max_images:
                break

            track = tracks_dict.get((track_id, video_id))
            video_path = video_paths.get(video_id)

            if track is None or video_path is None:
                continue

            # Get a frame from the middle of the track
            mid_idx = len(track.timestamps) // 2
            timestamp = track.timestamps[mid_idx]
            bbox = track.bboxes[mid_idx]

            # Extract frame
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num = int(timestamp * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                continue

            # Crop to person
            x, y, w, h = bbox
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            # Convert to PIL
            img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            images.append(img)
            labels.append(f"{video_id}\nTrack {track_id}")

        if not images:
            print("No images to visualize")
            return

        # Create grid
        from PIL import ImageDraw, ImageFont

        cols = min(5, len(images))
        rows = (len(images) + cols - 1) // cols

        # Resize to thumbnails
        thumb_size = (200, 300)
        thumbs = [img.resize(thumb_size) for img in images]

        # Create grid
        grid_w = cols * thumb_size[0]
        grid_h = rows * thumb_size[1]
        grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

        draw = ImageDraw.Draw(grid)

        for i, (thumb, label) in enumerate(zip(thumbs, labels)):
            col = i % cols
            row = i // cols
            x_pos = col * thumb_size[0]
            y_pos = row * thumb_size[1]

            grid.paste(thumb, (x_pos, y_pos))

            # Draw label
            draw.text((x_pos + 5, y_pos + 5), label, fill=(255, 0, 0))

        grid.save(output_path)

        person_name = cluster.person_name or f"Person {cluster_id}"
        print(f"✓ Saved visualization for {person_name} to {output_path}")
