"""
Face Detection and Embedding Extraction Module

This module provides face detection, recognition, and embedding extraction
for the hybrid video search system using InsightFace.
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
    Face detection and embedding extraction using InsightFace

    Supports:
    - Face detection with bounding boxes
    - Face recognition embeddings (512-dim)
    - Face attribute detection (age, gender, etc.)
    - Multiple faces per image
    """

    def __init__(self, model_name: str = "buffalo_l", device: str = "cuda",
                 det_size: Tuple[int, int] = (640, 640)):
        """
        Initialize face detection and recognition model

        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_s, antelopev2)
            device: Device to run on ("cuda" or "cpu")
            det_size: Detection input size (width, height)
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is required for face detection. "
                "Install with: pip install insightface onnxruntime-gpu"
            )

        self.device = device
        self.model_name = model_name
        self.det_size = det_size

        # Initialize face analysis app
        # Use GPU if available, otherwise CPU
        ctx_id = 0 if device == "cuda" else -1

        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda"
                     else ['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        print(f"✓ Initialized InsightFace model: {model_name}")

    def detect_faces(self, image: Union[str, np.ndarray, Image.Image],
                    min_confidence: float = 0.5) -> List[Dict]:
        """
        Detect faces in an image

        Args:
            image: Image path, numpy array, or PIL Image
            min_confidence: Minimum detection confidence

        Returns:
            List of face detections with bounding boxes, embeddings, and attributes
        """
        # Convert to numpy array
        img_array = self._to_numpy(image)

        # Detect faces
        faces = self.app.get(img_array)

        # Filter by confidence and format results
        detections = []
        for face in faces:
            if face.det_score >= min_confidence:
                detection = {
                    "bbox": face.bbox.astype(int).tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(face.det_score),
                    "embedding": face.normed_embedding.astype(np.float32),
                    "landmarks": face.kps.astype(int).tolist(),  # 5 facial landmarks
                }

                # Add attributes if available
                if hasattr(face, 'age'):
                    detection["age"] = int(face.age)
                if hasattr(face, 'gender'):
                    detection["gender"] = "male" if face.gender == 1 else "female"

                detections.append(detection)

        return detections

    def extract_face_embeddings(self, image: Union[str, np.ndarray, Image.Image],
                               min_confidence: float = 0.5) -> List[np.ndarray]:
        """
        Extract face embeddings from an image

        Args:
            image: Image path, numpy array, or PIL Image
            min_confidence: Minimum detection confidence

        Returns:
            List of face embeddings (512-dim normalized vectors)
        """
        detections = self.detect_faces(image, min_confidence)
        return [det["embedding"] for det in detections]

    def extract_largest_face_embedding(self, image: Union[str, np.ndarray, Image.Image],
                                      min_confidence: float = 0.5) -> Optional[np.ndarray]:
        """
        Extract embedding for the largest face in the image

        Args:
            image: Image path, numpy array, or PIL Image
            min_confidence: Minimum detection confidence

        Returns:
            Face embedding (512-dim) or None if no face found
        """
        detections = self.detect_faces(image, min_confidence)

        if not detections:
            return None

        # Find largest face (by bounding box area)
        largest_face = max(detections, key=lambda d: self._bbox_area(d["bbox"]))
        return largest_face["embedding"]

    def extract_face_crops(self, image: Union[str, np.ndarray, Image.Image],
                          min_confidence: float = 0.5,
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
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

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
                          min_confidence: float = 0.5,
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
            detections = self.detect_faces(image, min_confidence)
            all_detections.append(detections)

        return all_detections

    def visualize_faces(self, image: Union[str, np.ndarray, Image.Image],
                       detections: Optional[List[Dict]] = None,
                       min_confidence: float = 0.5,
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

            # Draw landmarks
            if "landmarks" in det:
                for lm in det["landmarks"]:
                    x, y = lm
                    draw.ellipse([x-2, y-2, x+2, y+2], fill="red")

            # Draw label
            label = f"Face {i+1}: {confidence:.2f}"
            if "age" in det:
                label += f"\nAge: {det['age']}"
            if "gender" in det:
                label += f" {det['gender']}"

            draw.text((x1, y1 - 40), label, fill="green", font=font)

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
                             min_confidence: float = 0.5):
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
            # Compute similarity
            similarity = np.dot(query_embedding, face["embedding"])

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
    """Example usage of face detection"""
    import argparse

    parser = argparse.ArgumentParser(description="Face Detection and Recognition")
    parser.add_argument("--image", type=str, help="Image path for face detection")
    parser.add_argument("--detect", action="store_true", help="Detect faces in image")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min confidence")
    parser.add_argument("--output", type=str, help="Output path for visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Initialize face embeddings
    print("Initializing face detection...")
    face_emb = FaceEmbeddings(device=args.device)

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

        # Visualize
        output_path = args.output or "face_detection_result.jpg"
        face_emb.visualize_faces(args.image, detections, output_path=output_path)


if __name__ == "__main__":
    main()
