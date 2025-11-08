"""
Face Detection and Embedding Extraction Module using DeepFace

This module provides face detection, recognition, and embedding extraction
for the hybrid video search system using DeepFace.

DeepFace supports multiple backends:
- VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID
- ArcFace, Dlib, SFace, GhostFaceNet
"""

import io
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


class FaceEmbeddings:
    """
    Face detection and embedding extraction using DeepFace

    Supports:
    - Face detection with bounding boxes
    - Face recognition embeddings (varies by model: 128-4096 dim)
    - Face attribute detection (age, gender, race, emotion)
    - Multiple faces per image
    - Multiple backend models
    """

    def __init__(self, model_name: str = "Facenet512",
                 detector_backend: str = "retinaface",
                 device: str = "cuda"):
        """
        Initialize face detection and recognition model

        Args:
            model_name: DeepFace model (VGG-Face, Facenet, Facenet512, ArcFace, etc.)
            detector_backend: Detection backend (opencv, ssd, dlib, mtcnn, retinaface, mediapipe, yolov8, yunet, fastmtcnn)
            device: Device to run on ("cuda" or "cpu")
        """
        try:
            from deepface import DeepFace
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "deepface is required for face detection. "
                "Install with: pip install deepface tf-keras"
            )

        self.DeepFace = DeepFace
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.device = device

        # Configure TensorFlow to use GPU if available
        if device == "cuda":
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✓ GPU acceleration enabled: {len(gpus)} GPU(s) found")
                except RuntimeError as e:
                    print(f"⚠ GPU configuration error: {e}")
        else:
            # Force CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Warm up the model by running a dummy detection
        try:
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            self.DeepFace.extract_faces(dummy_img, detector_backend=self.detector_backend)
            print(f"✓ Initialized DeepFace with {model_name} model and {detector_backend} detector")
        except Exception as e:
            print(f"⚠ Warning during initialization: {e}")

    def detect_faces(self, image: Union[str, np.ndarray, Image.Image],
                    min_confidence: float = 0.9,
                    align: bool = True) -> List[Dict]:
        """
        Detect faces in an image

        Args:
            image: Image path, numpy array, or PIL Image
            min_confidence: Minimum detection confidence (0-1)
            align: Whether to align faces

        Returns:
            List of face detections with bounding boxes, embeddings, and attributes
        """
        # Convert to numpy array
        img_array = self._to_numpy(image)

        try:
            # Extract faces with detection
            faces_data = self.DeepFace.extract_faces(
                img_array,
                detector_backend=self.detector_backend,
                align=align,
                enforce_detection=False
            )
        except Exception as e:
            warnings.warn(f"Face detection failed: {e}")
            return []

        detections = []

        for face_data in faces_data:
            confidence = face_data.get('confidence', 0.0)

            if confidence >= min_confidence:
                # Get facial area (bounding box)
                facial_area = face_data.get('facial_area', {})
                bbox = [
                    facial_area.get('x', 0),
                    facial_area.get('y', 0),
                    facial_area.get('x', 0) + facial_area.get('w', 0),
                    facial_area.get('y', 0) + facial_area.get('h', 0)
                ]

                detection = {
                    "bbox": bbox,  # [x1, y1, x2, y2]
                    "confidence": float(confidence),
                    "face_image": face_data.get('face'),  # Cropped and aligned face
                }

                # Store for embedding extraction later
                detection["_face_data"] = face_data

                detections.append(detection)

        # Extract embeddings for detected faces
        if detections:
            try:
                # Get embeddings for the original image
                embeddings = self.DeepFace.represent(
                    img_array,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    align=align
                )

                # Match embeddings to detections
                for i, detection in enumerate(detections):
                    if i < len(embeddings):
                        embedding = embeddings[i].get('embedding', [])
                        detection["embedding"] = np.array(embedding, dtype=np.float32)
                    else:
                        # Fallback: extract from cropped face
                        face_img = detection["_face_data"].get("face")
                        if face_img is not None:
                            try:
                                emb = self.DeepFace.represent(
                                    face_img,
                                    model_name=self.model_name,
                                    enforce_detection=False
                                )
                                detection["embedding"] = np.array(emb[0]["embedding"], dtype=np.float32)
                            except:
                                detection["embedding"] = None

                    # Clean up temporary data
                    del detection["_face_data"]

            except Exception as e:
                warnings.warn(f"Embedding extraction failed: {e}")
                # Remove embeddings from all detections
                for detection in detections:
                    detection["embedding"] = None
                    if "_face_data" in detection:
                        del detection["_face_data"]

        # Extract attributes (age, gender, etc.) if available
        try:
            attributes = self.DeepFace.analyze(
                img_array,
                actions=['age', 'gender', 'race', 'emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )

            # Match attributes to detections
            for i, detection in enumerate(detections):
                if i < len(attributes):
                    attr = attributes[i]
                    detection["age"] = int(attr.get('age', 0))
                    detection["gender"] = attr.get('dominant_gender', 'unknown')
                    detection["race"] = attr.get('dominant_race', 'unknown')
                    detection["emotion"] = attr.get('dominant_emotion', 'unknown')
        except Exception as e:
            # Attributes are optional
            pass

        return detections

    def extract_face_embeddings(self, image: Union[str, np.ndarray, Image.Image],
                               min_confidence: float = 0.9) -> List[np.ndarray]:
        """
        Extract face embeddings from an image

        Args:
            image: Image path, numpy array, or PIL Image
            min_confidence: Minimum detection confidence

        Returns:
            List of face embeddings
        """
        detections = self.detect_faces(image, min_confidence)
        embeddings = []

        for det in detections:
            if det.get("embedding") is not None:
                embeddings.append(det["embedding"])

        return embeddings

    def extract_largest_face_embedding(self, image: Union[str, np.ndarray, Image.Image],
                                      min_confidence: float = 0.9) -> Optional[np.ndarray]:
        """
        Extract embedding for the largest face in the image

        Args:
            image: Image path, numpy array, or PIL Image
            min_confidence: Minimum detection confidence

        Returns:
            Face embedding or None if no face found
        """
        detections = self.detect_faces(image, min_confidence)

        if not detections:
            return None

        # Find largest face (by bounding box area)
        largest_face = max(detections, key=lambda d: self._bbox_area(d["bbox"]))
        return largest_face.get("embedding")

    def detect_faces_from_image(self, image: Image.Image,
                               min_confidence: float = 0.9) -> List[Dict]:
        """
        Detect faces from a PIL Image (alias for detect_faces for clarity)

        Args:
            image: PIL Image object
            min_confidence: Minimum detection confidence

        Returns:
            List of face detections
        """
        return self.detect_faces(image, min_confidence)

    def extract_face_crops(self, image: Union[str, np.ndarray, Image.Image],
                          min_confidence: float = 0.9,
                          padding: int = 20) -> List[Tuple[Image.Image, Dict]]:
        """
        Extract cropped face images with their metadata

        Args:
            image: Image path, numpy array, or PIL Image
            min_confidence: Minimum detection confidence
            padding: Padding around face bounding box

        Returns:
            List of (cropped_face_image, detection_info) tuples
        """
        detections = self.detect_faces(image, min_confidence)

        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image.convert('RGB')

        crops = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.width, x2 + padding)
            y2 = min(img.height, y2 + padding)

            # Crop face
            face_crop = img.crop((x1, y1, x2, y2))
            crops.append((face_crop, det))

        return crops

    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Normalize embeddings
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)

        # Convert to 0-1 range
        similarity = (similarity + 1) / 2

        return float(similarity)

    def verify_faces(self, img1: Union[str, np.ndarray],
                    img2: Union[str, np.ndarray]) -> Dict:
        """
        Verify if two images contain the same person using DeepFace.verify

        Args:
            img1: First image
            img2: Second image

        Returns:
            Dict with verification result and similarity metrics
        """
        try:
            result = self.DeepFace.verify(
                img1_path=self._to_path_or_array(img1),
                img2_path=self._to_path_or_array(img2),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )

            return {
                "verified": result.get("verified", False),
                "distance": result.get("distance", 1.0),
                "threshold": result.get("threshold", 0.0),
                "similarity": 1.0 / (1.0 + result.get("distance", 1.0))  # Convert distance to similarity
            }
        except Exception as e:
            warnings.warn(f"Face verification failed: {e}")
            return {
                "verified": False,
                "distance": 1.0,
                "threshold": 0.0,
                "similarity": 0.0
            }

    def identify_face(self, query_embedding: np.ndarray,
                     known_embeddings: List[np.ndarray],
                     threshold: float = 0.6) -> Optional[int]:
        """
        Identify a face by comparing to known embeddings

        Args:
            query_embedding: Query face embedding
            known_embeddings: List of known face embeddings
            threshold: Similarity threshold for positive match

        Returns:
            Index of matched face or None if no match
        """
        best_match_idx = None
        best_similarity = threshold

        for idx, known_emb in enumerate(known_embeddings):
            similarity = self.compare_faces(query_embedding, known_emb)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = idx

        return best_match_idx

    def batch_detect_faces(self, images: List[Union[str, np.ndarray, Image.Image]],
                          min_confidence: float = 0.9,
                          show_progress: bool = True) -> List[List[Dict]]:
        """
        Detect faces in multiple images

        Args:
            images: List of images
            min_confidence: Minimum detection confidence
            show_progress: Show progress bar

        Returns:
            List of face detection lists (one per image)
        """
        all_detections = []

        iterator = tqdm(images, desc="Detecting faces") if show_progress else images

        for image in iterator:
            try:
                detections = self.detect_faces(image, min_confidence)
                all_detections.append(detections)
            except Exception as e:
                warnings.warn(f"Failed to process image: {e}")
                all_detections.append([])

        return all_detections

    def visualize_faces(self, image: Union[str, np.ndarray, Image.Image],
                       detections: Optional[List[Dict]] = None,
                       min_confidence: float = 0.9,
                       output_path: Optional[str] = None) -> Image.Image:
        """
        Visualize face detections on image

        Args:
            image: Image to visualize
            detections: Pre-computed detections (or None to compute)
            min_confidence: Minimum confidence if computing detections
            output_path: Optional path to save visualization

        Returns:
            PIL Image with visualized detections
        """
        # Load image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image.convert('RGB')

        # Get detections if not provided
        if detections is None:
            detections = self.detect_faces(image, min_confidence)

        # Draw detections
        draw = ImageDraw.Draw(img)

        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

            # Draw label
            label = f"Face {i+1}: {confidence:.2f}"
            if "age" in det:
                label += f"\nAge: {det['age']}"
            if "gender" in det:
                label += f" {det['gender']}"
            if "emotion" in det:
                label += f"\n{det['emotion']}"

            draw.text((x1, y1 - 60), label, fill="green", font=font)

        if output_path:
            img.save(output_path)
            print(f"✓ Saved visualization to {output_path}")

        return img

    @staticmethod
    def _to_numpy(image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to numpy array (RGB format)"""
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
            return np.array(img)
        elif isinstance(image, Image.Image):
            return np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            # Ensure RGB
            if len(image.shape) == 2:  # Grayscale
                return np.stack([image] * 3, axis=-1)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _to_path_or_array(image: Union[str, np.ndarray]) -> Union[str, np.ndarray]:
        """Convert image to path or array for DeepFace"""
        if isinstance(image, str):
            return image
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _bbox_area(bbox: List[int]) -> int:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)


class FaceDatabase:
    """
    Face database for storing and searching known faces
    """

    def __init__(self):
        """Initialize face database"""
        self.faces = []  # List of {"name": str, "embedding": np.ndarray, "metadata": dict}

    def add_face(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        """
        Add a face to the database

        Args:
            name: Person's name or identifier
            embedding: Face embedding
            metadata: Optional metadata (image path, timestamp, etc.)
        """
        self.faces.append({
            "name": name,
            "embedding": embedding,
            "metadata": metadata or {}
        })

    def add_faces_from_images(self, name: str, image_paths: List[str],
                             face_embeddings: 'FaceEmbeddings',
                             min_confidence: float = 0.9):
        """
        Add multiple face images for a person

        Args:
            name: Person's name
            image_paths: List of image paths containing the person
            face_embeddings: FaceEmbeddings instance
            min_confidence: Minimum detection confidence
        """
        for img_path in image_paths:
            embedding = face_embeddings.extract_largest_face_embedding(
                img_path, min_confidence
            )
            if embedding is not None:
                self.add_face(name, embedding, {"source_image": img_path})
            else:
                warnings.warn(f"No face detected in {img_path}")

    def search(self, query_embedding: np.ndarray, threshold: float = 0.6,
              top_k: int = 5) -> List[Dict]:
        """
        Search for matching faces

        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold
            top_k: Number of top results to return

        Returns:
            List of matches with scores
        """
        if not self.faces:
            return []

        results = []

        for face in self.faces:
            # Compute similarity using cosine similarity
            face_emb = face["embedding"]

            # Normalize
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            face_norm = face_emb / (np.linalg.norm(face_emb) + 1e-8)

            # Cosine similarity
            similarity = np.dot(query_norm, face_norm)
            similarity = (similarity + 1) / 2  # Convert to 0-1 range

            if similarity >= threshold:
                results.append({
                    "name": face["name"],
                    "similarity": float(similarity),
                    "metadata": face["metadata"]
                })

        # Sort by similarity
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]

    def identify(self, query_embedding: np.ndarray, threshold: float = 0.6) -> Optional[str]:
        """
        Identify a face (return best match name or None)

        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold

        Returns:
            Name of best match or None
        """
        results = self.search(query_embedding, threshold, top_k=1)
        return results[0]["name"] if results else None

    def save(self, path: str):
        """Save face database to file"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.faces, f)
        print(f"✓ Saved face database with {len(self.faces)} faces to {path}")

    def load(self, path: str):
        """Load face database from file"""
        import pickle
        with open(path, 'rb') as f:
            self.faces = pickle.load(f)
        print(f"✓ Loaded face database with {len(self.faces)} faces from {path}")

    def get_all_names(self) -> List[str]:
        """Get list of all unique names in database"""
        return list(set(face["name"] for face in self.faces))

    def get_faces_by_name(self, name: str) -> List[Dict]:
        """Get all faces for a specific person"""
        return [face for face in self.faces if face["name"] == name]

    def remove_face(self, name: str):
        """Remove all faces for a person"""
        self.faces = [face for face in self.faces if face["name"] != name]

    def __len__(self):
        return len(self.faces)


def main():
    """Example usage of face detection with DeepFace"""
    import argparse

    parser = argparse.ArgumentParser(description="Face Detection and Recognition with DeepFace")
    parser.add_argument("--image", type=str, help="Image path for face detection")
    parser.add_argument("--detect", action="store_true", help="Detect faces in image")
    parser.add_argument("--min-confidence", type=float, default=0.9, help="Min confidence")
    parser.add_argument("--output", type=str, help="Output path for visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--model", type=str, default="Facenet512",
                       help="DeepFace model (VGG-Face, Facenet, Facenet512, ArcFace, etc.)")
    parser.add_argument("--detector", type=str, default="retinaface",
                       help="Detector backend (opencv, ssd, dlib, mtcnn, retinaface, etc.)")

    args = parser.parse_args()

    # Initialize face embeddings
    print("Initializing face detection with DeepFace...")
    face_emb = FaceEmbeddings(
        model_name=args.model,
        detector_backend=args.detector,
        device=args.device
    )

    if args.detect and args.image:
        print(f"\nDetecting faces in {args.image}...")
        detections = face_emb.detect_faces(args.image, args.min_confidence)

        print(f"Found {len(detections)} face(s):")
        for i, det in enumerate(detections, 1):
            print(f"\nFace {i}:")
            print(f"  Confidence: {det['confidence']:.3f}")
            print(f"  BBox: {det['bbox']}")
            if "age" in det:
                print(f"  Age: {det['age']}")
            if "gender" in det:
                print(f"  Gender: {det['gender']}")
            if "emotion" in det:
                print(f"  Emotion: {det['emotion']}")

        # Visualize
        output_path = args.output or "face_detection_result.jpg"
        face_emb.visualize_faces(args.image, detections, output_path=output_path)


if __name__ == "__main__":
    main()
