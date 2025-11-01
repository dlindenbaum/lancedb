"""
Advanced Example: Object Localization

This example demonstrates how to find and localize specific objects
within video frames using text-guided dense features.
"""

from video_search import HybridVideoSearch
import os
from pathlib import Path


def main():
    print("="*60)
    print("Object Localization Example")
    print("="*60)

    # Initialize search system
    search = HybridVideoSearch(
        db_path="./video_search_demo.db",
        device="cuda"  # Use "cpu" if no GPU
    )

    # Frame to analyze
    frame_path = "path/to/frame.jpg"

    if not os.path.exists(frame_path):
        print(f"⚠ Frame not found at {frame_path}")
        print("Please provide a valid frame path")
        return

    # Example 1: Detect objects with different queries
    print("\nExample 1: Multi-Query Object Detection")
    print("-" * 60)

    queries = [
        "American flag",
        "person's face",
        "car",
        "tree",
        "building"
    ]

    for query in queries:
        print(f"\nSearching for: '{query}'")

        # Find objects
        detections = search.locate_object(
            frame_path=frame_path,
            query=query,
            threshold=0.25  # Adjust threshold based on confidence needed
        )

        if detections:
            print(f"  Found {len(detections)} detection(s):")
            for i, det in enumerate(detections, 1):
                x, y, w, h = det["bbox"]
                score = det["score"]
                print(f"    {i}. BBox: ({x}, {y}, {w}, {h}), Score: {score:.3f}")

            # Visualize detections
            output_path = f"detection_{query.replace(' ', '_')}.jpg"
            search.visualize_detections(
                frame_path=frame_path,
                detections=detections,
                output_path=output_path
            )
            print(f"  ✓ Visualization saved to: {output_path}")
        else:
            print(f"  No detections above threshold")

    # Example 2: Multi-object detection
    print("\n" + "="*60)
    print("Example 2: Multiple Objects in Single Frame")
    print("-" * 60)

    # Detect multiple objects at once
    multi_queries = [
        ("person", "red"),
        ("flag", "red"),
        ("car", "blue")
    ]

    all_detections = {}

    for obj, color in multi_queries:
        query = f"{color} {obj}"
        print(f"\nDetecting: '{query}'")

        detections = search.locate_object(
            frame_path=frame_path,
            query=query,
            threshold=0.2
        )

        all_detections[query] = detections

        if detections:
            print(f"  Found {len(detections)} instance(s)")
        else:
            print(f"  None found")

    # Example 3: Fine-grained search with low threshold
    print("\n" + "="*60)
    print("Example 3: Fine-Grained Search (Low Threshold)")
    print("-" * 60)

    fine_queries = [
        "logo",
        "text",
        "button"
    ]

    for query in fine_queries:
        print(f"\nSearching for: '{query}' (low threshold)")

        detections = search.locate_object(
            frame_path=frame_path,
            query=query,
            threshold=0.15  # Lower threshold for harder-to-find objects
        )

        print(f"  Found {len(detections)} detection(s)")

        if detections:
            # Show top 3 detections
            for i, det in enumerate(sorted(detections, key=lambda x: x["score"], reverse=True)[:3], 1):
                x, y, w, h = det["bbox"]
                print(f"    {i}. BBox: ({x}, {y}, {w}, {h}), Score: {det['score']:.3f}")

    # Example 4: Batch processing on multiple frames
    print("\n" + "="*60)
    print("Example 4: Batch Processing Multiple Frames")
    print("-" * 60)

    frames_dir = "path/to/frames_directory"

    if os.path.exists(frames_dir):
        frame_files = sorted(Path(frames_dir).glob("*.jpg"))[:5]  # Process first 5 frames

        target_object = "person"
        print(f"Searching for '{target_object}' in {len(frame_files)} frames...")

        for frame_file in frame_files:
            print(f"\nFrame: {frame_file.name}")

            detections = search.locate_object(
                frame_path=str(frame_file),
                query=target_object,
                threshold=0.25
            )

            if detections:
                print(f"  ✓ Found {len(detections)} detection(s)")

                # Save visualization
                output_path = f"detected_{frame_file.stem}.jpg"
                search.visualize_detections(
                    frame_path=str(frame_file),
                    detections=detections,
                    output_path=output_path
                )
            else:
                print(f"  ✗ No detections")
    else:
        print(f"⚠ Frames directory not found: {frames_dir}")

    # Example 5: Temporal object tracking
    print("\n" + "="*60)
    print("Example 5: Temporal Object Tracking")
    print("-" * 60)

    print("Searching for object across video timeline...")

    # First, search for frames containing the object
    query = "person wearing red"
    results = search.search_text(query, limit=20)

    if results:
        print(f"Found {len(results)} frames with '{query}'")

        # Now localize the object in each frame
        tracking_results = []

        for result in results[:5]:  # Process top 5 results
            frame_path = result['frame_path']
            timestamp = result['timestamp']

            detections = search.locate_object(
                frame_path=frame_path,
                query=query,
                threshold=0.25
            )

            if detections:
                tracking_results.append({
                    'timestamp': timestamp,
                    'frame_id': result['frame_id'],
                    'detections': detections
                })

        # Display tracking timeline
        print("\nTracking Timeline:")
        for track in tracking_results:
            print(f"  Time: {track['timestamp']:.2f}s, Frame: {track['frame_id']}, "
                  f"Detections: {len(track['detections'])}")

    print("\n" + "="*60)
    print("Object Localization Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
