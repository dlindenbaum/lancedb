# Usage Guide: Hybrid Video Search

This guide provides detailed usage examples for the hybrid video search system.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Common Use Cases](#common-use-cases)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

## Installation

### Option 1: Using pip

```bash
cd python/examples/multimodal_video_search
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
conda create -n video_search python=3.9
conda activate video_search
pip install -r requirements.txt
```

### Verify Installation

```python
from video_search import HybridVideoSearch
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
search = HybridVideoSearch(db_path="test.db")
print("✓ Installation successful!")
```

## Quick Start

### 1. Index Your First Video

```python
from video_search import HybridVideoSearch

# Initialize
search = HybridVideoSearch(db_path="my_videos.db")

# Index a video
search.index_video(
    video_path="path/to/video.mp4",
    video_id="vid001",
    fps=2  # Extract 2 frames per second
)
```

### 2. Search with Text

```python
# Find frames matching text description
results = search.search_text("person wearing red shirt", limit=10)

for result in results:
    print(f"Frame {result['frame_id']} at {result['timestamp']:.2f}s")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Path: {result['frame_path']}")
```

### 3. Search with Image

```python
# Find visually similar frames
results = search.search_image("reference_image.jpg", limit=10)
```

## Core Concepts

### Embedding Types

The system uses two types of embeddings:

#### 1. CLIP Embeddings (Semantic)
- Best for: Text-to-image search, semantic understanding
- Use case: "Find frames with American flag"
- Dimension: 512 (clip_vit_b32) or 512 (clip_vit_b16)
- **Uses ONNX Runtime for efficient inference**

#### 2. DINOv3 Embeddings (Visual)
- Best for: Visual similarity, fine-grained features
- Use case: Find frames that look exactly like a reference
- Dimension: 384 (dinov2_vits14) or 768 (dinov2_vitb14)
- **Uses ONNX Runtime for efficient inference**

### Search Modes

#### Text Search
```python
# Semantic search using CLIP
results = search.search_text("outdoor scene with trees")
```

#### Image Search
```python
# Visual similarity using CLIP + DINOv3
results = search.search_image(
    "query.jpg",
    clip_weight=0.5,   # Semantic similarity
    dino_weight=0.5    # Visual similarity
)
```

#### Hybrid Search
```python
# Combine text and image
results = search.hybrid_search(
    text="American flag",
    image="flag_example.jpg",
    text_weight=0.6,
    image_weight=0.4
)
```

## Common Use Cases

### Use Case 1: Content Moderation

```python
# Initialize
search = HybridVideoSearch(db_path="content_moderation.db")

# Index all videos
for video_file in video_files:
    search.index_video(video_file, video_id=video_file, fps=1)

# Search for problematic content
flagged = search.search_text("violence", limit=100)

# Review with hybrid search
suspicious = search.hybrid_search(
    text="weapon",
    image="reference_weapon.jpg",
    limit=50
)
```

### Use Case 2: Surveillance Video Search

```python
# Index surveillance footage
for cam_id, video_path in cameras.items():
    search.index_video(
        video_path,
        video_id=f"camera_{cam_id}",
        fps=0.5  # Lower FPS for surveillance
    )

# Search for specific events
results = search.search_text("person entering through door")

# Localize person in frame
for result in results[:10]:
    detections = search.locate_object(
        result['frame_path'],
        query="person",
        threshold=0.25
    )

    # Save detection visualization
    if detections:
        search.visualize_detections(
            result['frame_path'],
            detections,
            output_path=f"detection_{result['frame_id']}.jpg"
        )
```

### Use Case 3: Sports Video Analysis

```python
# Index game footage
search.index_video("game_footage.mp4", "game_001", fps=3)

# Find specific plays
touchdowns = search.search_text("touchdown celebration", limit=20)

# Find similar plays
similar_plays = search.search_image("example_touchdown.jpg", limit=10)

# Track player across frames
player_frames = search.search_text("player number 23", limit=50)

for frame in player_frames[:10]:
    # Localize player
    detections = search.locate_object(
        frame['frame_path'],
        query="player number 23",
        threshold=0.2
    )
```

### Use Case 4: Marketing & Ad Analysis

```python
# Index commercial footage
search.index_video("commercials.mp4", "ads_2024", fps=2)

# Find brand appearances
brand_results = search.search_text("coca cola logo", limit=100)

# Find product placements
product_results = search.hybrid_search(
    text="smartphone",
    image="iphone_reference.jpg",
    text_weight=0.4,
    image_weight=0.6
)

# Measure screen time
total_appearances = len(brand_results)
total_seconds = sum(1/2 for _ in brand_results)  # fps=2
print(f"Brand appeared in {total_appearances} frames ({total_seconds:.1f}s)")
```

## Performance Optimization

### GPU Acceleration

```python
# Use CUDA for 10-100x speedup
search = HybridVideoSearch(
    db_path="videos.db",
    device="cuda",
    batch_size=64  # Increase for faster GPUs
)
```

### Batch Processing

```python
# Process multiple videos in parallel
from concurrent.futures import ThreadPoolExecutor

def index_video_wrapper(video_info):
    video_path, video_id = video_info
    return search.index_video(video_path, video_id, fps=2)

with ThreadPoolExecutor(max_workers=4) as executor:
    video_infos = [(path, vid_id) for path, vid_id in videos.items()]
    results = executor.map(index_video_wrapper, video_infos)
```

### Frame Rate Selection

```python
# Adjust FPS based on content type

# Fast action (sports, surveillance): 2-5 fps
search.index_video(video, "action_vid", fps=3)

# Slow content (interviews, static scenes): 0.5-1 fps
search.index_video(video, "interview_vid", fps=0.5)

# Detailed analysis: 5-10 fps
search.index_video(video, "detailed_vid", fps=10)
```

### Model Selection

```python
# Fast (CLIP ViT-B/32): ~100ms per image
search = HybridVideoSearch(
    clip_model="clip_vit_b32",
    dino_model="dinov2_vits14"
)

# More accurate (CLIP ViT-B/16): ~150ms per image
search = HybridVideoSearch(
    clip_model="clip_vit_b16",
    dino_model="dinov2_vitb14"
)

# Fast indexing (CLIP only)
search = HybridVideoSearch(
    use_dino=False  # 2x faster indexing
)
```

### Memory Management

```python
# Reduce memory usage

# Smaller batch size
search = HybridVideoSearch(batch_size=16)

# Process in chunks
search.index_video(
    video_path,
    video_id,
    max_frames=1000  # Process 1000 frames at a time
)

# Clean up frames after indexing
search.index_video(
    video_path,
    video_id,
    cleanup_frames=True  # Delete frames after embedding
)
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1: Reduce batch size**
```python
search = HybridVideoSearch(batch_size=8)
```

**Solution 2: Use CPU**
```python
search = HybridVideoSearch(device="cpu")
```

**Solution 3: Disable DINOv3**
```python
search = HybridVideoSearch(use_dino=False)
```

### Issue: Slow Indexing

**Solution 1: Use faster CLIP model**
```python
search = HybridVideoSearch(clip_model="clip_vit_b32")
```

**Solution 2: Reduce FPS**
```python
search.index_video(video, "vid", fps=1)  # Instead of fps=2
```

**Solution 3: Skip DINOv3**
```python
search = HybridVideoSearch(use_dino=False)
```

### Issue: Poor Search Results

**Solution 1: Adjust weights**
```python
# For semantic search
results = search.search_image(img, clip_weight=0.7, dino_weight=0.3)

# For visual similarity
results = search.search_image(img, clip_weight=0.3, dino_weight=0.7)
```

**Solution 2: Use hybrid search**
```python
# Combine text and image
results = search.hybrid_search(
    text="American flag",
    image="flag_example.jpg"
)
```

**Solution 3: Try different CLIP model**
```python
# Larger model = better accuracy
search = HybridVideoSearch(clip_model="clip_vit_b16")
```

### Issue: No Detections in Object Localization

**Solution 1: Lower threshold**
```python
detections = search.locate_object(
    frame_path,
    query="flag",
    threshold=0.15  # Lower from 0.25
)
```

**Solution 2: Use more specific query**
```python
# Instead of "flag"
detections = search.locate_object(
    frame_path,
    query="American flag with red and white stripes",
    threshold=0.25
)
```

**Solution 3: Try different queries**
```python
queries = ["American flag", "flag", "stars and stripes"]
all_detections = []

for query in queries:
    dets = search.locate_object(frame_path, query, threshold=0.2)
    all_detections.extend(dets)

# Remove duplicates with NMS
final_detections = search._nms(all_detections, iou_threshold=0.5)
```

## Advanced Features

### Custom Embedding Functions

```python
from lancedb.embeddings import OpenClipEmbeddings

# Use custom CLIP model
custom_clip = OpenClipEmbeddings(
    name="ViT-L-14",
    pretrained="laion2b_s32b_b82k",
    device="cuda"
)

search.clip_embeddings = custom_clip
```

### Multi-Table Search

```python
# Access tables directly
clip_table = search.db.open_table("clip_embeddings")
dino_table = search.db.open_table("dino_embeddings")

# Custom queries
clip_results = clip_table.search(query_vec).limit(20).to_pandas()
dino_results = dino_table.search(query_vec).limit(20).to_pandas()
```

### Export to DataFrame

```python
import pandas as pd

# Get all indexed frames
clip_table = search.db.open_table("clip_embeddings")
df = clip_table.to_pandas()

# Analyze
print(f"Total frames: {len(df)}")
print(f"Videos: {df['video_id'].nunique()}")
print(f"Duration: {df['timestamp'].max():.1f}s")

# Export
df.to_csv("indexed_frames.csv", index=False)
```

## Best Practices

1. **Index organization**: Use meaningful video_ids (e.g., "cam1_20240101", "interview_john_smith")
2. **FPS selection**: Balance between coverage and storage (2 fps is usually good)
3. **Model choice**: Start with ViT-B/32, upgrade to ViT-L/14 if needed
4. **Threshold tuning**: Start with 0.25, adjust based on precision/recall needs
5. **Batch processing**: Index multiple videos in parallel for large datasets
6. **Storage**: Clean up frames after indexing to save disk space
7. **Backup**: Regularly backup your LanceDB database

## Next Steps

- Explore the [example_basic.py](example_basic.py) for complete examples
- Check [example_object_detection.py](example_object_detection.py) for localization
- Try the [Jupyter notebook](example_notebook.ipynb) for interactive exploration
- Read the main [README.md](README.md) for API reference
