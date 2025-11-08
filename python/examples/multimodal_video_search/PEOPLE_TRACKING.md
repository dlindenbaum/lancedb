# People Tracking with YOLO + Roboflow Supervision

This document provides detailed information about the people tracking capabilities of the video search system.

## Overview

The people tracking system uses:
- **YOLOv8 (ONNX)** for person detection
- **Roboflow Supervision (ByteTrack)** for multi-object tracking
- **LanceDB** for storing and querying track data

## Features

- ✅ **Real-time person detection** using YOLOv8 ONNX models
- ✅ **Multi-person tracking** with ByteTrack algorithm
- ✅ **Persistent track IDs** across frames
- ✅ **Track analytics** (duration, frequency, movement patterns)
- ✅ **Database storage** in LanceDB for efficient queries
- ✅ **Visualization** of person tracks
- ✅ **Track filtering** by duration, detections, etc.

## Installation

The people tracking feature requires the `supervision` library:

```bash
pip install supervision>=0.16.0
```

All other dependencies (ONNX Runtime, etc.) are already included in the main requirements.

## Quick Start

### 1. Initialize with People Tracking

```python
from video_search import HybridVideoSearch

# Enable people tracking
search = HybridVideoSearch(
    db_path="tracking.db",
    use_people_tracking=True,  # Enable YOLO + ByteTrack
    yolo_conf_threshold=0.5,   # Confidence threshold (0-1)
    device="cuda"              # Use GPU for faster processing
)
```

### 2. Track People in a Video

```python
# Track all people throughout the video
tracks = search.track_people_in_video(
    video_path="path/to/video.mp4",
    video_id="video_001",
    fps=2,  # Process 2 frames per second
    max_frames=None,  # Process all frames
    save_tracks=True  # Save to database
)

# Each track contains:
for track in tracks:
    print(f"Person ID: {track.track_id}")
    print(f"  Duration: {track.duration():.2f}s")
    print(f"  Appearances: {track.num_detections()} frames")
    print(f"  First seen: {track.first_timestamp:.2f}s")
    print(f"  Last seen: {track.last_timestamp:.2f}s")
```

### 3. Query Tracked People

```python
# Get all tracked people
all_people = search.get_tracked_people(video_id="video_001")

# Filter by minimum duration
long_appearances = search.get_tracked_people(
    video_id="video_001",
    min_duration=5.0,  # At least 5 seconds
    min_detections=10  # At least 10 frame detections
)

# Find brief appearances
brief_appearances = search.get_tracked_people(
    video_id="video_001",
    min_duration=0.0,
    min_detections=1
)
brief_only = [t for t in brief_appearances if t['duration'] < 1.0]
```

### 4. Get Person Timeline

```python
# Get detailed information for a specific person
timeline = search.get_person_timeline(track_id=1, video_id="video_001")

print(f"Person {timeline['track_id']}:")
print(f"  Duration: {timeline['duration']:.2f}s")
print(f"  Frames: {timeline['frame_ids']}")
print(f"  Timestamps: {timeline['timestamps']}")
print(f"  Bounding boxes: {timeline['bboxes']}")
print(f"  Confidences: {timeline['confidences']}")
```

### 5. Visualize Tracks

```python
# Create a grid visualization showing a person across multiple frames
search.visualize_person_track(
    track_id=1,
    video_id="video_001",
    video_path="path/to/video.mp4",
    output_path="person_1_track.jpg",
    max_frames=10  # Show up to 10 frames
)
```

## Track Data Structure

Each `PersonTrack` object contains:

```python
@dataclass
class PersonTrack:
    track_id: int                    # Unique ID for this person
    video_id: str                    # Video identifier
    first_frame: int                 # First frame number where person appears
    last_frame: int                  # Last frame number where person appears
    first_timestamp: float           # First timestamp (seconds)
    last_timestamp: float            # Last timestamp (seconds)
    bboxes: List[List[float]]       # Bounding boxes [x, y, w, h] for each frame
    frame_ids: List[int]            # Frame IDs where person was detected
    timestamps: List[float]         # Timestamps for each detection
    confidences: List[float]        # Detection confidence scores
    appearance_embedding: np.ndarray # Optional: Appearance feature vector
    face_embedding: np.ndarray      # Optional: Face embedding (if face detection enabled)
    person_id: str                  # Optional: Identified person name/ID
```

## Statistics and Analytics

### Get Tracking Statistics

```python
stats = search.get_people_tracking_statistics(video_id="video_001")

print(f"Total unique people: {stats['total_tracks']}")
print(f"Average duration: {stats['avg_duration']:.2f}s")
print(f"Max duration: {stats['max_duration']:.2f}s")
print(f"Min duration: {stats['min_duration']:.2f}s")
print(f"Average detections: {stats['avg_detections']:.1f}")
print(f"Max detections: {stats['max_detections']}")
```

### Export Track Data

```python
from people_tracking import PeopleTracker

# Export tracks to JSON
tracker = PeopleTracker(device="cpu")
tracker.tracks = {t.track_id: t for t in tracks}
tracker.export_tracks("tracks.json")
```

## Advanced Usage

### Combine with Face Recognition

You can combine people tracking with face recognition to identify specific individuals:

```python
# Enable both tracking and face recognition
search = HybridVideoSearch(
    db_path="tracking.db",
    use_people_tracking=True,
    use_face_detection=True,  # Also enable face recognition
    face_model="Facenet512",
    yolo_conf_threshold=0.5
)

# Track people and detect faces
tracks = search.track_people_in_video(
    video_path="video.mp4",
    video_id="video_001",
    fps=2
)

# Index faces separately
search.index_video_faces(
    video_path="video.mp4",
    video_id="video_001",
    fps=2
)

# Now you can correlate tracks with face embeddings
# by matching frame IDs and timestamps
```

### Custom YOLO Model

You can use your own custom YOLOv8 ONNX model:

```python
from people_tracking import PeopleTracker

# Use custom YOLO model
tracker = PeopleTracker(
    yolo_model_path="/path/to/custom_yolo.onnx",
    conf_threshold=0.6,
    device="cuda"
)

# Track video
tracks = tracker.track_video(
    video_path="video.mp4",
    video_id="custom_001",
    fps=2
)
```

### Adjust Detection Parameters

```python
# More conservative detection (fewer false positives)
search = HybridVideoSearch(
    use_people_tracking=True,
    yolo_conf_threshold=0.7,  # Higher confidence threshold
    device="cuda"
)

# More aggressive detection (catch more people, may have false positives)
search = HybridVideoSearch(
    use_people_tracking=True,
    yolo_conf_threshold=0.3,  # Lower confidence threshold
    device="cuda"
)
```

## Use Cases

### 1. Surveillance Analysis

```python
# Track all people in surveillance footage
search = HybridVideoSearch(use_people_tracking=True)

# Process multiple cameras
for cam_id in ["cam1", "cam2", "cam3"]:
    tracks = search.track_people_in_video(
        video_path=f"footage/{cam_id}.mp4",
        video_id=cam_id,
        fps=1  # Lower FPS for long surveillance videos
    )

# Find people appearing across multiple cameras
# (requires face recognition for person re-identification)
```

### 2. Crowd Analytics

```python
# Track people in a crowded scene
tracks = search.track_people_in_video(
    video_path="crowd_video.mp4",
    video_id="crowd_001",
    fps=5  # Higher FPS for accurate crowd tracking
)

# Analyze crowd density over time
from collections import Counter

frame_counts = Counter()
for track in tracks:
    for frame_id in track.frame_ids:
        frame_counts[frame_id] += 1

# Find peak crowd density
peak_frame = max(frame_counts, key=frame_counts.get)
peak_count = frame_counts[peak_frame]
print(f"Peak crowd: {peak_count} people at frame {peak_frame}")
```

### 3. Retail Analytics

```python
# Track customer movement in store
tracks = search.track_people_in_video(
    video_path="store_footage.mp4",
    video_id="store_001",
    fps=2
)

# Find customers who spent the most time
long_visitors = search.get_tracked_people(
    video_id="store_001",
    min_duration=60.0,  # At least 1 minute
    min_detections=20
)

print(f"Long-duration visitors: {len(long_visitors)}")

# Find quick pass-throughs
quick_visitors = [t for t in tracks if t.duration() < 10.0]
print(f"Quick pass-throughs: {len(quick_visitors)}")
```

### 4. Sports Analytics

```python
# Track players in a sports video
tracks = search.track_people_in_video(
    video_path="game.mp4",
    video_id="game_001",
    fps=10  # Higher FPS for fast movement
)

# Analyze player movement patterns
for track in tracks:
    # Calculate average position
    avg_x = sum(bbox[0] + bbox[2]/2 for bbox in track.bboxes) / len(track.bboxes)
    avg_y = sum(bbox[1] + bbox[3]/2 for bbox in track.bboxes) / len(track.bboxes)

    print(f"Player {track.track_id} average position: ({avg_x:.0f}, {avg_y:.0f})")
```

## Performance Considerations

### Frame Rate Selection

- **Low FPS (1-2)**: Faster processing, good for surveillance or long videos
- **Medium FPS (5-10)**: Balanced, good for most use cases
- **High FPS (15-30)**: Slower but more accurate tracking, good for fast movement

### GPU vs CPU

```python
# GPU (faster, recommended)
search = HybridVideoSearch(use_people_tracking=True, device="cuda")

# CPU (slower but works without GPU)
search = HybridVideoSearch(use_people_tracking=True, device="cpu")
```

### Batch Processing

For processing multiple videos efficiently:

```python
search = HybridVideoSearch(use_people_tracking=True, device="cuda")

video_files = [
    ("video1.mp4", "vid_001"),
    ("video2.mp4", "vid_002"),
    ("video3.mp4", "vid_003"),
]

for video_path, video_id in video_files:
    print(f"Processing {video_id}...")
    tracks = search.track_people_in_video(
        video_path=video_path,
        video_id=video_id,
        fps=2,
        save_tracks=True
    )
    print(f"  Found {len(tracks)} people")
```

## Troubleshooting

### Issue: No People Detected

**Solutions:**
1. Lower the confidence threshold: `yolo_conf_threshold=0.3`
2. Ensure video contains people and is not too low quality
3. Check that the YOLO model downloaded correctly

### Issue: Too Many False Positives

**Solutions:**
1. Increase confidence threshold: `yolo_conf_threshold=0.7`
2. Post-process to filter out short tracks: `min_detections=5`

### Issue: Slow Processing

**Solutions:**
1. Use lower FPS: `fps=1`
2. Ensure GPU is being used: `device="cuda"`
3. Process subset of video: `max_frames=100`
4. Use lighter YOLO model (YOLOv8n is used by default)

### Issue: Lost Tracks (IDs switching)

ByteTrack is designed to handle occlusions and re-identify people. However, in challenging scenarios:

**Solutions:**
1. Increase FPS for better temporal continuity
2. Enable face recognition for better re-identification
3. Reduce `iou_threshold` in ByteTrack (modify in `people_tracking.py`)

## Model Information

### YOLOv8 ONNX Models

The system automatically downloads YOLOv8n (nano) ONNX model from HuggingFace:
- **Model**: YOLOv8n
- **Input size**: 640x640
- **Speed**: ~3-5ms on GPU
- **Accuracy**: mAP 37.3 on COCO

You can export custom YOLOv8 models to ONNX using:

```bash
# Install ultralytics
pip install ultralytics

# Export to ONNX
yolo export model=yolov8n.pt format=onnx
```

### ByteTrack Algorithm

ByteTrack is a simple yet effective multi-object tracking algorithm that:
- Associates detections across frames based on IoU
- Handles occlusions by maintaining tracks through missing detections
- Re-identifies people after temporary occlusions
- Assigns consistent IDs throughout the video

## Database Schema

Tracks are stored in the `people_tracks` table with the following schema:

```
people_tracks
├── track_id: int            # Unique track ID
├── video_id: str            # Video identifier
├── first_frame: int         # First frame number
├── last_frame: int          # Last frame number
├── first_timestamp: float   # First timestamp (seconds)
├── last_timestamp: float    # Last timestamp (seconds)
├── duration: float          # Total duration (seconds)
├── num_detections: int      # Number of frame detections
├── bboxes: List[List[float]]    # Bounding boxes
├── frame_ids: List[int]     # Frame IDs
├── timestamps: List[float]  # Timestamps
├── confidences: List[float] # Detection confidences
└── person_id: str           # Optional identified person
```

## Examples

See `example_people_tracking.py` for a complete working example.

## References

- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **ByteTrack**: https://github.com/ifzhang/ByteTrack
- **Roboflow Supervision**: https://github.com/roboflow/supervision
- **ONNX Runtime**: https://onnxruntime.ai/
