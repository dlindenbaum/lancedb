# Face Detection and Recognition Guide

This guide covers face detection, recognition, and tracking features in the hybrid video search system.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Face Detection](#face-detection)
5. [Face Search](#face-search)
6. [Person Tracking](#person-tracking)
7. [Face Database](#face-database)
8. [Advanced Features](#advanced-features)
9. [Use Cases](#use-cases)

## Overview

The face detection module uses **DeepFace** for:
- High-accuracy face detection with multiple backends
- Face embeddings (128-4096 dim depending on model)
- Face attribute detection (age, gender, race, emotion)
- Multi-face detection per frame
- Real-time performance on GPU

### Model Options

DeepFace supports multiple recognition models:
- **Facenet512** (default): 512-dim embeddings, good accuracy
- **ArcFace**: State-of-the-art accuracy
- **VGG-Face**: Classic model, 2622-dim embeddings
- **Facenet**: 128-dim embeddings, fast
- **OpenFace**: Lightweight, 128-dim
- **DeepID**: Fast inference
- **Dlib**: Good for CPU

### Detector Backends

- **retinaface** (default): Best accuracy
- **mtcnn**: Good balance
- **opencv**: Fastest, lower accuracy
- **ssd**: Fast and accurate
- **dlib**: CPU-friendly
- **mediapipe**: Mobile-optimized
- **yolov8**: Latest YOLO

## Installation

```bash
# Install face detection dependencies
pip install deepface tf-keras

# For GPU (recommended for faster processing)
pip install tensorflow[and-cuda]

# For CPU-only systems
pip install tensorflow
```

## Quick Start

### Initialize with Face Detection

```python
from video_search import HybridVideoSearch

# Enable face detection
search = HybridVideoSearch(
    db_path="video_search.db",
    use_face_detection=True,
    face_model="Facenet512",  # Good accuracy and speed (default)
    device="cuda"
)
```

### Index Faces from Video

```python
# Extract and index faces
num_faces = search.index_video_faces(
    video_path="video.mp4",
    video_id="surveillance_001",
    fps=2,  # Process 2 frames per second
    min_confidence=0.5  # Minimum face detection confidence
)

print(f"Indexed {num_faces} faces")
```

### Search for a Person

```python
# Find all appearances of a person
results = search.search_faces(
    query_face="person_photo.jpg",
    similarity_threshold=0.6,  # 60% similarity
    limit=20
)

for result in results:
    print(f"Found at {result['timestamp']:.2f}s - Similarity: {result['similarity']:.3f}")
```

## Face Detection

### Detect Faces in Single Frame

```python
# Detect faces
detections = search.detect_faces_in_frame(
    frame_path="frame.jpg",
    min_confidence=0.5,
    visualize=True,
    output_path="detected_faces.jpg"
)

# Access detection info
for face in detections:
    print(f"Face confidence: {face['confidence']:.3f}")
    print(f"Bounding box: {face['bbox']}")
    print(f"Age: {face.get('age', 'N/A')}")
    print(f"Gender: {face.get('gender', 'N/A')}")
```

### Extract Face Crops

```python
# Extract individual face images
crops = search.extract_face_crops(
    frame_path="frame.jpg",
    min_confidence=0.5,
    output_dir="face_crops"  # Save crops here
)

# Process each face crop
for face_img, detection in crops:
    # face_img is a PIL Image
    print(f"Face size: {face_img.size}")
    print(f"Confidence: {detection['confidence']:.3f}")
```

### Batch Face Detection

```python
from face_embeddings import FaceEmbeddings

face_detector = FaceEmbeddings(device="cuda")

# Detect faces in multiple images
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
all_detections = face_detector.batch_detect_faces(
    images,
    min_confidence=0.5,
    show_progress=True
)

# Process results
for img_path, detections in zip(images, all_detections):
    print(f"{img_path}: {len(detections)} face(s)")
```

## Face Search

### Basic Face Search

```python
# Search by reference image
results = search.search_faces(
    query_face="reference.jpg",
    similarity_threshold=0.6,
    limit=10
)

# Or search by embedding (if you have it)
import numpy as np
query_embedding = np.array([...])  # 512-dim embedding
results = search.search_faces(
    query_face=query_embedding,
    similarity_threshold=0.6,
    limit=10
)
```

### Search Results

```python
for result in results:
    print(f"Video: {result['video_id']}")
    print(f"Frame: {result['frame_id']} at {result['timestamp']:.2f}s")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Face location: {result['bbox']}")

    # Optional attributes
    if 'age' in result:
        print(f"Estimated age: {result['age']}")
    if 'gender' in result:
        print(f"Gender: {result['gender']}")
```

### Adjust Similarity Threshold

```python
# High precision (fewer results, but more accurate)
results = search.search_faces(
    query_face="person.jpg",
    similarity_threshold=0.75,  # Strict matching
    limit=10
)

# High recall (more results, but may include false positives)
results = search.search_faces(
    query_face="person.jpg",
    similarity_threshold=0.5,  # Looser matching
    limit=50
)
```

## Person Tracking

### Track Person Across Video

```python
# Find all appearances of a person in video
appearances = search.find_person_across_video(
    reference_face="person.jpg",
    video_id="video_001",
    similarity_threshold=0.6,
    limit=100
)

# Results are sorted by timestamp
print(f"Found {len(appearances)} appearances:")
for app in appearances:
    print(f"  Time: {app['timestamp']:6.2f}s | "
          f"Similarity: {app['similarity']:.3f}")
```

### Tracking Statistics

```python
if appearances:
    # Compute screen time
    first_time = appearances[0]['timestamp']
    last_time = appearances[-1]['timestamp']

    print(f"First appearance: {first_time:.2f}s")
    print(f"Last appearance: {last_time:.2f}s")
    print(f"Active duration: {last_time - first_time:.2f}s")
    print(f"Number of detections: {len(appearances)}")
```

### Visualize Tracking Timeline

```python
import matplotlib.pyplot as plt

# Extract timestamps and similarities
times = [app['timestamp'] for app in appearances]
similarities = [app['similarity'] for app in appearances]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(times, similarities, 'b-o')
plt.xlabel('Time (seconds)')
plt.ylabel('Face Similarity Score')
plt.title('Person Tracking Over Time')
plt.grid(True)
plt.show()
```

## Face Database

### Create Face Database for Known People

```python
from face_embeddings import FaceDatabase

# Initialize database
face_db = FaceDatabase()

# Add known people
known_people = {
    "Alice Johnson": ["alice1.jpg", "alice2.jpg", "alice3.jpg"],
    "Bob Smith": ["bob1.jpg", "bob2.jpg"],
    "Charlie Brown": ["charlie1.jpg"]
}

# Add faces to database
for name, image_paths in known_people.items():
    face_db.add_faces_from_images(
        name=name,
        image_paths=image_paths,
        face_embeddings=search.face_embeddings,
        min_confidence=0.5
    )

print(f"Database has {len(face_db)} face entries")
print(f"Known people: {face_db.get_all_names()}")
```

### Search in Face Database

```python
# Identify person in image
query_embedding = search.face_embeddings.extract_largest_face_embedding("unknown.jpg")

# Search database
matches = face_db.search(
    query_embedding,
    threshold=0.6,
    top_k=3
)

if matches:
    print("Possible matches:")
    for match in matches:
        print(f"  {match['name']}: {match['similarity']:.3f}")
else:
    print("No matches found")
```

### Save and Load Database

```python
# Save database
face_db.save("known_faces.pkl")

# Load database
new_db = FaceDatabase()
new_db.load("known_faces.pkl")
```

## Advanced Features

### Face Statistics and Analytics

```python
# Get overall statistics
stats = search.get_face_statistics()

print(f"Total faces indexed: {stats['total_faces']}")
print(f"Unique frames with faces: {stats['unique_frames']}")
print(f"Average faces per frame: {stats['avg_faces_per_frame']:.2f}")

# Gender distribution
if 'gender_distribution' in stats:
    print("\nGender distribution:")
    for gender, count in stats['gender_distribution'].items():
        percentage = (count / stats['total_faces']) * 100
        print(f"  {gender}: {count} ({percentage:.1f}%)")

# Age statistics
if 'age_mean' in stats:
    print(f"\nAge statistics:")
    print(f"  Minimum: {stats['age_min']}")
    print(f"  Maximum: {stats['age_max']}")
    print(f"  Average: {stats['age_mean']:.1f}")
```

### Per-Video Statistics

```python
# Get statistics for specific video
video_stats = search.get_face_statistics(video_id="video_001")

print(f"Video: {video_stats['videos'][0]}")
print(f"  Total faces: {video_stats['total_faces']}")
print(f"  Unique frames: {video_stats['unique_frames']}")
print(f"  Avg faces/frame: {video_stats['avg_faces_per_frame']:.2f}")
```

### Compare Two Faces

```python
# Extract embeddings
emb1 = search.face_embeddings.extract_largest_face_embedding("person1.jpg")
emb2 = search.face_embeddings.extract_largest_face_embedding("person2.jpg")

# Compare
similarity = search.face_embeddings.compare_faces(emb1, emb2)

if similarity > 0.6:
    print(f"Same person (similarity: {similarity:.3f})")
else:
    print(f"Different people (similarity: {similarity:.3f})")
```

### Dense Face Grid Extraction

```python
# Process entire directory
from pathlib import Path

images_dir = Path("images/")
face_counts = {}

for img_file in images_dir.glob("*.jpg"):
    detections = search.detect_faces_in_frame(str(img_file))
    face_counts[img_file.name] = len(detections)

# Find images with most faces
sorted_images = sorted(face_counts.items(), key=lambda x: x[1], reverse=True)
print("Images with most faces:")
for img_name, count in sorted_images[:10]:
    print(f"  {img_name}: {count} faces")
```

## Use Cases

### Use Case 1: Security and Surveillance

```python
# Index surveillance footage
search = HybridVideoSearch(db_path="surveillance.db", use_face_detection=True)

# Index multiple camera feeds
for cam_id in ["cam1", "cam2", "cam3"]:
    video_path = f"footage/{cam_id}.mp4"
    search.index_video_faces(video_path, video_id=cam_id, fps=1)

# Search for person of interest
poi_photo = "person_of_interest.jpg"
sightings = search.search_faces(poi_photo, similarity_threshold=0.7, limit=100)

print(f"Found {len(sightings)} potential sightings")
for sighting in sightings:
    print(f"  Camera: {sighting['video_id']}, Time: {sighting['timestamp']:.2f}s")
```

### Use Case 2: Media and Entertainment

```python
# Index movie/TV content
search.index_video_faces("movie.mp4", "movie_001", fps=1)

# Find scenes with specific actor
actor_photo = "actor_reference.jpg"
scenes = search.find_person_across_video(
    actor_photo,
    video_id="movie_001",
    similarity_threshold=0.6
)

# Compute screen time
screen_time = len(scenes) * (1/1)  # fps=1
print(f"Actor screen time: {screen_time:.1f} seconds")
```

### Use Case 3: Demographic Analysis

```python
# Index promotional video
search.index_video_faces("promo_video.mp4", "promo_001", fps=2)

# Get demographics
stats = search.get_face_statistics("promo_001")

print("Video Demographics:")
print(f"  Total people shown: {stats['total_faces']}")
print(f"  Gender distribution: {stats['gender_distribution']}")
print(f"  Age range: {stats['age_min']}-{stats['age_max']}")
print(f"  Average age: {stats['age_mean']:.1f}")
```

### Use Case 4: Event Photography

```python
# Process event photos
event_photos = list(Path("event_photos/").glob("*.jpg"))

# Detect all faces
all_faces = []
for photo in event_photos:
    detections = search.detect_faces_in_frame(str(photo))
    all_faces.extend(detections)

print(f"Total faces detected: {len(all_faces)}")

# Find photos of specific person
guest_photo = "guest_reference.jpg"
guest_photos = []

for photo in event_photos:
    results = search.search_faces(guest_photo, limit=1)
    if results and results[0]['frame_path'] == str(photo):
        guest_photos.append(photo)

print(f"Found guest in {len(guest_photos)} photos")
```

## Performance Tips

### GPU Acceleration

```python
# Use GPU for faster face detection
search = HybridVideoSearch(
    use_face_detection=True,
    device="cuda"  # 10-100x faster than CPU
)
```

### Batch Processing

```python
# Process multiple videos in parallel
from concurrent.futures import ThreadPoolExecutor

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]

def process_video(video_path):
    video_id = Path(video_path).stem
    return search.index_video_faces(video_path, video_id, fps=1)

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_video, videos))

print(f"Total faces indexed: {sum(results)}")
```

### Optimize Detection Speed

```python
# Use smaller/faster model for faster processing
search = HybridVideoSearch(
    use_face_detection=True,
    face_model="Facenet",  # Faster than Facenet512 (128-dim vs 512-dim)
    device="cuda"
)

# Or use OpenFace for lightweight processing
search = HybridVideoSearch(
    use_face_detection=True,
    face_model="OpenFace",  # Lightweight, good for CPU
    device="cpu"
)

# Reduce FPS for faster indexing
search.index_video_faces(video_path, video_id, fps=0.5)  # 1 frame every 2 seconds
```

## Troubleshooting

### Issue: No Faces Detected

**Solution 1: Lower confidence threshold**
```python
detections = search.detect_faces_in_frame(
    frame_path,
    min_confidence=0.3  # Lower from 0.5
)
```

**Solution 2: Check image quality**
- Ensure faces are clearly visible
- Minimum face size: ~30x30 pixels
- Good lighting and no heavy occlusion

### Issue: Too Many False Positives

**Solution: Increase similarity threshold**
```python
results = search.search_faces(
    query_face,
    similarity_threshold=0.75  # Increase from 0.6
)
```

### Issue: Slow Performance

**Solution 1: Use GPU**
```python
search = HybridVideoSearch(device="cuda", use_face_detection=True)
```

**Solution 2: Use smaller/faster model**
```python
# Use Facenet (128-dim, faster) instead of Facenet512 (512-dim)
search = HybridVideoSearch(face_model="Facenet", use_face_detection=True)

# Or use OpenFace for lightweight processing
search = HybridVideoSearch(face_model="OpenFace", use_face_detection=True)
```

**Solution 3: Reduce frame rate**
```python
search.index_video_faces(video_path, video_id, fps=0.5)
```

## Best Practices

1. **Confidence Threshold**: Start with 0.5, adjust based on precision/recall needs
2. **Similarity Threshold**: Use 0.6-0.7 for most applications
3. **Frame Rate**:
   - Surveillance: 0.5-1 fps
   - Action videos: 2-3 fps
   - High-quality analysis: 5+ fps
4. **Model Selection**:
   - Facenet512: Good accuracy, 512-dim (default)
   - ArcFace: Best accuracy
   - Facenet: Fast, 128-dim
   - OpenFace: Lightweight, good for CPU
   - VGG-Face: Classic, very high dimensional
5. **Database Management**: Regularly save face databases to avoid recomputation

## Next Steps

- Try the [example_face_search.py](example_face_search.py) for complete examples
- Explore the main [README.md](README.md) for the full system
- Check [USAGE.md](USAGE.md) for general usage guide
