# Hybrid Video Search with CLIP + DINOv3 + Face Recognition + People Tracking + Person Clustering + LanceDB

A comprehensive multimodal video search system that combines:
- **CLIP (ONNX)**: Semantic text-to-image search ("American flag in background", "person wearing red")
- **DINOv3 (ONNX)**: Fine-grained visual features for precise object localization
- **Face Recognition**: Detect, recognize, and track faces using DeepFace
- **People Tracking**: Track and identify people using YOLO (ONNX) + Roboflow Supervision
- **Person Clustering**: Identify unique individuals across tracks and videos using face clustering
- **LanceDB**: High-performance vector storage and retrieval

**🚀 All models use ONNX Runtime for efficient, lightweight inference without PyTorch dependency!**

## Features

- 🔍 **Text-to-Video Search**: Find frames using natural language queries
- 🎯 **Object Localization**: Locate specific objects within frames
- 🖼️ **Image Similarity**: Find visually similar frames
- 🎨 **Hybrid Search**: Combine text and image queries for better results
- 👤 **Face Detection**: Detect and extract faces from video frames
- 🔎 **Face Search**: Find specific people across videos
- 📊 **Face Analytics**: Age, gender, and demographic analysis
- 🚶 **People Tracking**: Track people across video with YOLO + ByteTrack
- 🎭 **Person Identification**: Assign unique IDs to each person and track movement
- 🧑‍🤝‍🧑 **Person Clustering**: Identify same person across different tracks using face recognition
- 🔍 **Person Re-ID**: Re-identify people across multiple videos
- 🏷️ **Person Naming**: Assign names to identified individuals
- 📈 **Track Analytics**: Duration, frequency, and movement patterns
- ⚡ **Fast & Scalable**: Built on LanceDB for efficient vector search
- 🪶 **Lightweight**: ONNX models for efficient inference without heavy ML frameworks

## Architecture

```
┌──────────────┬──────────┬─────────────┬──────────────┬──────────────┬─────────────┐
│   Model      │  Text    │   Visual    │  Faces       │  Tracking    │  Best For   │
│              │  Search  │  Precision  │              │              │             │
├──────────────┼──────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ CLIP         │   ✓✓✓    │      ✓      │      ✗       │      ✗       │ Semantic    │
│              │          │             │              │              │ search      │
├──────────────┼──────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ DINOv3       │   ✗      │     ✓✓✓     │      ✗       │      ✗       │ Visual      │
│              │          │             │              │              │ similarity  │
├──────────────┼──────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ DeepFace     │   ✗      │      ✗      │     ✓✓✓      │      ✗       │ Face        │
│              │          │             │              │              │ recognition │
├──────────────┼──────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ YOLO +       │   ✗      │      ✗      │      ✗       │     ✓✓✓      │ Person      │
│ ByteTrack    │          │             │              │              │ tracking    │
├──────────────┼──────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ All Combined │   ✓✓✓    │     ✓✓✓     │     ✓✓✓      │     ✓✓✓      │ Complete    │
└──────────────┴──────────┴─────────────┴──────────────┴──────────────┴─────────────┘
```

## Installation

```bash
cd python/examples/multimodal_video_search
pip install -r requirements.txt
```

## Quick Start

### 1. Index a Video

```python
from video_search import HybridVideoSearch

# Initialize the search system
search = HybridVideoSearch(db_path="video_search.db")

# Index a video (extracts frames at 2 fps)
search.index_video("path/to/video.mp4", video_id="video_001", fps=2)
```

### 2. Search with Text

```python
# Find frames with text query
results = search.search_text("American flag in the background", limit=10)

for result in results:
    print(f"Frame {result['frame_id']} at {result['timestamp']}s - Score: {result['score']:.3f}")
    # Display or save the frame
```

### 3. Search with Image

```python
# Find similar frames
results = search.search_image("path/to/query_image.jpg", limit=10)
```

### 4. Hybrid Search (Text + Image)

```python
# Combine text and image queries
results = search.hybrid_search(
    text="person wearing red shirt",
    image="path/to/example.jpg",
    text_weight=0.6,
    image_weight=0.4,
    limit=10
)
```

### 5. Object Localization

```python
# Find and localize specific objects in a frame
detections = search.locate_object(
    frame_path="frame.jpg",
    query="American flag",
    threshold=0.25
)

# Visualize detections
search.visualize_detections("frame.jpg", detections, output_path="output.jpg")
```

### 6. Face Detection and Search

```python
# Enable face detection
search = HybridVideoSearch(
    db_path="video_search.db",
    use_face_detection=True  # Enable face recognition
)

# Index faces from video
num_faces = search.index_video_faces(
    video_path="video.mp4",
    video_id="video_001",
    fps=2,
    min_confidence=0.5
)

# Search for similar faces
results = search.search_faces(
    query_face="person_photo.jpg",
    similarity_threshold=0.6,
    limit=10
)

# Track a person across video
appearances = search.find_person_across_video(
    reference_face="person_photo.jpg",
    video_id="video_001",
    similarity_threshold=0.6
)

# Get face statistics
stats = search.get_face_statistics()
print(f"Total faces: {stats['total_faces']}")
print(f"Gender distribution: {stats['gender_distribution']}")
print(f"Average age: {stats['age_mean']:.1f}")
```

### 7. People Tracking (YOLO + ByteTrack)

```python
# Enable people tracking
search = HybridVideoSearch(
    db_path="video_search.db",
    use_people_tracking=True,  # Enable YOLO + ByteTrack
    yolo_conf_threshold=0.5,   # Detection confidence threshold
    use_face_detection=True    # Optional: combine with face recognition
)

# Track all people in a video
tracks = search.track_people_in_video(
    video_path="video.mp4",
    video_id="video_001",
    fps=2,  # Process 2 frames per second
    save_tracks=True  # Save to database
)

print(f"Found {len(tracks)} unique people")

# Get tracked people from database
all_people = search.get_tracked_people(video_id="video_001")

# Filter by appearance duration
prominent_people = search.get_tracked_people(
    video_id="video_001",
    min_duration=5.0,  # At least 5 seconds
    min_detections=10  # At least 10 frames
)

# Get detailed timeline for a specific person
timeline = search.get_person_timeline(track_id=1, video_id="video_001")
print(f"Person appeared for {timeline['duration']:.2f}s")
print(f"Detected in {timeline['num_detections']} frames")

# Visualize a person's track across frames
search.visualize_person_track(
    track_id=1,
    video_id="video_001",
    video_path="video.mp4",
    output_path="person_track.jpg",
    max_frames=10
)

# Get tracking statistics
stats = search.get_people_tracking_statistics(video_id="video_001")
print(f"Total unique people: {stats['total_tracks']}")
print(f"Average duration: {stats['avg_duration']:.2f}s")
print(f"Longest appearance: {stats['max_duration']:.2f}s")
```

### 8. Person Clustering (Identify same person across tracks/videos)

```python
# Enable both tracking and face recognition
search = HybridVideoSearch(
    db_path="video_search.db",
    use_people_tracking=True,  # Enable tracking
    use_face_detection=True,   # Enable face recognition
    face_model="Facenet512"
)

# Track people in multiple videos
for video_id, video_path in [("vid1", "v1.mp4"), ("vid2", "v2.mp4")]:
    search.track_people_in_video(
        video_path=video_path,
        video_id=video_id,
        fps=2
    )

# Cluster tracked people to identify unique individuals
video_paths = {"vid1": "v1.mp4", "vid2": "v2.mp4"}
clusters = search.cluster_tracked_people(
    video_ids=["vid1", "vid2"],
    video_paths=video_paths,
    similarity_threshold=0.6,  # Face similarity threshold
    clustering_method="dbscan",  # or "agglomerative"
    save_clusters=True
)

print(f"Found {len(clusters)} unique people")

# Assign names to identified people
search.assign_person_names({
    0: "John Doe",
    1: "Jane Smith",
    2: "Bob Johnson"
})

# Search for a specific person by name
appearances = search.search_for_person("John Doe")
print(f"John Doe appears in {len(appearances)} tracks")

# Identify a person from a photo
result = search.identify_person_from_image(
    query_image="person_photo.jpg",
    min_similarity=0.6
)

if result:
    print(f"Matched: {result['person_name']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Appears in {result['num_videos']} videos")

# Get all appearances of a person
person_appearances = search.get_person_appearances(cluster_id=0)

# Visualize a person's appearances across videos
search.visualize_person_cluster(
    cluster_id=0,
    output_path="person_0_grid.jpg",
    video_paths=video_paths,
    max_images=10
)

# Get clustering statistics
stats = search.get_clustering_statistics()
print(f"Unique people: {stats['num_clusters']}")
print(f"Avg tracks per person: {stats['avg_tracks_per_person']:.1f}")
print(f"Avg videos per person: {stats['avg_videos_per_person']:.1f}")
```

## Usage Examples

### Example 1: Search Surveillance Footage

```python
# Index multiple surveillance videos
search = HybridVideoSearch(db_path="surveillance.db")

for video_file in ["cam1.mp4", "cam2.mp4", "cam3.mp4"]:
    search.index_video(video_file, video_id=video_file, fps=1)

# Search for specific events
results = search.search_text("person entering through door", limit=20)
```

### Example 2: Content Moderation

```python
# Search for specific content
flagged_content = search.search_text("inappropriate content", limit=100)

# Combine with image example
suspicious = search.hybrid_search(
    text="prohibited item",
    image="examples/prohibited_item.jpg",
    text_weight=0.5,
    image_weight=0.5
)
```

### Example 3: Sports Highlights

```python
# Find specific plays
touchdowns = search.search_text("football touchdown celebration", limit=50)

# Find similar plays to a reference image
similar_plays = search.search_image("example_touchdown.jpg", limit=20)
```

## API Reference

### HybridVideoSearch

#### `__init__(db_path, clip_model="ViT-B/32", device="cuda")`
Initialize the search system.

**Parameters:**
- `db_path` (str): Path to LanceDB database
- `clip_model` (str): CLIP model variant
- `device` (str): "cuda" or "cpu"

#### `index_video(video_path, video_id, fps=2)`
Extract and index frames from a video.

**Parameters:**
- `video_path` (str): Path to video file
- `video_id` (str): Unique identifier for the video
- `fps` (int): Frame extraction rate

#### `search_text(query, limit=10)`
Search using text query.

**Parameters:**
- `query` (str): Text description
- `limit` (int): Number of results

**Returns:** List of results with frame info and scores

#### `search_image(image_path, limit=10, clip_weight=0.5, dino_weight=0.5)`
Search using image query.

**Parameters:**
- `image_path` (str): Path to query image
- `limit` (int): Number of results
- `clip_weight` (float): Weight for CLIP similarity
- `dino_weight` (float): Weight for DINOv3 similarity

#### `hybrid_search(text, image, text_weight=0.5, image_weight=0.5, limit=10)`
Combined text and image search.

#### `locate_object(frame_path, query, threshold=0.25)`
Find and localize objects in a frame.

**Parameters:**
- `frame_path` (str): Path to frame image
- `query` (str): Text description of object to find
- `threshold` (float): Detection confidence threshold

**Returns:** List of detections with bounding boxes and scores

## Performance Tips

1. **GPU Acceleration**: Use CUDA for 10-100x speedup
2. **Batch Processing**: Process multiple videos in parallel
3. **Frame Sampling**: Adjust FPS based on video content (action: 2-5 fps, static: 0.5-1 fps)
4. **Model Selection**:
   - ViT-B/32: Fastest, good for most use cases
   - ViT-L/14: Better accuracy, slower
5. **Index Optimization**: Use LanceDB's IVF-PQ for large datasets

## Advanced Features

### Dense Feature Extraction

```python
# Extract patch-level features for fine-grained localization
from video_search import DenseFeatureExtractor

extractor = DenseFeatureExtractor()
patch_features, grid_size = extractor.extract_dense_clip(frame_path)
similarity_map = extractor.compute_similarity_map(patch_features, text_query)
```

### Multi-Table Search

```python
# Separate tables for different embedding types
search = HybridVideoSearch(db_path="search.db")

# Query both CLIP and DINOv3 tables
clip_results = search.db.open_table("clip_embeddings").search(query_vec).limit(20)
dino_results = search.db.open_table("dino_embeddings").search(query_vec).limit(20)

# Combine with custom weighting
combined = search.combine_results(clip_results, dino_results, weights=[0.6, 0.4])
```

### ONNX Model Options

```python
# Fast: CLIP ViT-B/32 (~100ms per image)
search = HybridVideoSearch(clip_model="clip_vit_b32", dino_model="dinov2_vits14")

# More accurate: CLIP ViT-B/16 (~150ms per image)
search = HybridVideoSearch(clip_model="clip_vit_b16", dino_model="dinov2_vitb14")
```

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
search = HybridVideoSearch(db_path="search.db", batch_size=16)

# Process videos in smaller chunks
search.index_video(video_path, fps=1, max_frames=1000)
```

### Slow Indexing

```python
# Use faster CLIP model (default)
search = HybridVideoSearch(clip_model="clip_vit_b32")

# Reduce frame rate
search.index_video(video_path, fps=0.5)

# Skip DINOv3 for faster indexing (text search only)
search = HybridVideoSearch(use_dino=False)
```

## License

Apache-2.0

## Citation

If you use this in your research, please cite:

```bibtex
@software{lancedb_video_search,
  title={Hybrid Video Search with CLIP and DINOv3},
  author={LanceDB Team},
  year={2024},
  url={https://github.com/lancedb/lancedb}
}
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](../../../CONTRIBUTING.md).

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
