"""
Face Search Examples

This example demonstrates:
1. Indexing faces from videos
2. Searching for similar faces
3. Tracking people across frames
4. Face detection and visualization
"""

from video_search import HybridVideoSearch
from face_embeddings import FaceDatabase
import os
from pathlib import Path


def main():
    print("="*60)
    print("Face Search Examples")
    print("="*60)

    # Initialize search system with face detection enabled
    print("\nInitializing video search with face detection...")
    search = HybridVideoSearch(
        db_path="./face_search_demo.db",
        device="cuda",  # Use "cpu" if no GPU
        use_face_detection=True,  # Enable face detection
        face_model="buffalo_l"  # High accuracy model
    )

    # Example 1: Index faces from a video
    print("\n" + "="*60)
    print("Example 1: Indexing Faces from Video")
    print("="*60)

    video_path = "path/to/video.mp4"

    if os.path.exists(video_path):
        # Index faces
        num_faces = search.index_video_faces(
            video_path=video_path,
            video_id="demo_video",
            fps=2,  # Process 2 frames per second
            min_confidence=0.5
        )
        print(f"✓ Indexed {num_faces} faces")
    else:
        print(f"⚠ Video not found at {video_path}")
        print("Skipping face indexing example")

    # Example 2: Detect faces in a single frame
    print("\n" + "="*60)
    print("Example 2: Detect Faces in a Frame")
    print("="*60)

    frame_path = "path/to/frame.jpg"

    if os.path.exists(frame_path):
        detections = search.detect_faces_in_frame(
            frame_path=frame_path,
            min_confidence=0.5,
            visualize=True,
            output_path="face_detections.jpg"
        )

        print(f"Found {len(detections)} face(s):")
        for i, det in enumerate(detections, 1):
            print(f"\nFace {i}:")
            print(f"  Confidence: {det['confidence']:.3f}")
            print(f"  BBox: {det['bbox']}")
            if 'age' in det:
                print(f"  Age: {det['age']}")
            if 'gender' in det:
                print(f"  Gender: {det['gender']}")
    else:
        print(f"⚠ Frame not found at {frame_path}")

    # Example 3: Search for similar faces
    print("\n" + "="*60)
    print("Example 3: Search for Similar Faces")
    print("="*60)

    reference_image = "path/to/person.jpg"

    if os.path.exists(reference_image):
        print(f"Searching for faces similar to {reference_image}...")

        results = search.search_faces(
            query_face=reference_image,
            similarity_threshold=0.6,  # 60% similarity
            limit=10
        )

        if results:
            print(f"Found {len(results)} matching face(s):")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Video: {result['video_id']}")
                print(f"   Frame: {result['frame_id']} at {result['timestamp']:.2f}s")
                print(f"   Similarity: {result['similarity']:.3f}")
                print(f"   BBox: {result['bbox']}")
                if 'age' in result:
                    print(f"   Age: ~{result['age']}")
                if 'gender' in result:
                    print(f"   Gender: {result['gender']}")
        else:
            print("No matching faces found (database might be empty)")
    else:
        print(f"⚠ Reference image not found at {reference_image}")

    # Example 4: Track a person across video
    print("\n" + "="*60)
    print("Example 4: Track Person Across Video")
    print("="*60)

    if os.path.exists(reference_image):
        print("Tracking person through video timeline...")

        # Find all appearances of the person
        appearances = search.find_person_across_video(
            reference_face=reference_image,
            video_id="demo_video",  # Search in specific video
            similarity_threshold=0.6,
            limit=50
        )

        if appearances:
            print(f"\nFound person in {len(appearances)} frame(s):")
            print("\nTimeline:")
            for app in appearances[:10]:  # Show first 10
                print(f"  Time: {app['timestamp']:6.2f}s | "
                      f"Frame: {app['frame_id']:4d} | "
                      f"Similarity: {app['similarity']:.3f}")

            # Compute statistics
            if len(appearances) > 1:
                total_time = appearances[-1]['timestamp'] - appearances[0]['timestamp']
                print(f"\nStatistics:")
                print(f"  First appearance: {appearances[0]['timestamp']:.2f}s")
                print(f"  Last appearance: {appearances[-1]['timestamp']:.2f}s")
                print(f"  Total screen time: ~{total_time:.2f}s")
        else:
            print("Person not found in video")
    else:
        print("Skipping tracking example (reference image not found)")

    # Example 5: Extract face crops
    print("\n" + "="*60)
    print("Example 5: Extract Face Crops")
    print("="*60)

    if os.path.exists(frame_path):
        print(f"Extracting face crops from {frame_path}...")

        crops = search.extract_face_crops(
            frame_path=frame_path,
            min_confidence=0.5,
            output_dir="face_crops"
        )

        if crops:
            print(f"✓ Extracted {len(crops)} face crop(s)")
            for i, (crop_img, det) in enumerate(crops, 1):
                print(f"  Crop {i}: {crop_img.size}, confidence: {det['confidence']:.3f}")
        else:
            print("No faces found in frame")
    else:
        print("Skipping face crops example")

    # Example 6: Face statistics
    print("\n" + "="*60)
    print("Example 6: Face Statistics")
    print("="*60)

    # Overall stats
    stats = search.get_face_statistics()
    print("\nOverall Statistics:")
    print(f"  Total faces indexed: {stats['total_faces']}")
    print(f"  Unique frames with faces: {stats['unique_frames']}")
    print(f"  Average faces per frame: {stats['avg_faces_per_frame']:.2f}")

    if 'gender_distribution' in stats:
        print(f"\n  Gender distribution:")
        for gender, count in stats['gender_distribution'].items():
            print(f"    {gender}: {count}")

    if 'age_mean' in stats:
        print(f"\n  Age statistics:")
        print(f"    Min: {stats['age_min']}")
        print(f"    Max: {stats['age_max']}")
        print(f"    Average: {stats['age_mean']:.1f}")

    # Per-video stats
    if stats['videos']:
        print(f"\n  Videos: {', '.join(stats['videos'])}")

        for video_id in stats['videos']:
            video_stats = search.get_face_statistics(video_id)
            print(f"\n  Stats for {video_id}:")
            print(f"    Faces: {video_stats['total_faces']}")
            print(f"    Frames: {video_stats['unique_frames']}")
            print(f"    Avg faces/frame: {video_stats['avg_faces_per_frame']:.2f}")

    # Example 7: Build a face database
    print("\n" + "="*60)
    print("Example 7: Face Database for Known People")
    print("="*60)

    face_db = FaceDatabase()

    # Add known people (if images exist)
    known_people = {
        "Alice": ["alice1.jpg", "alice2.jpg"],
        "Bob": ["bob1.jpg", "bob2.jpg"],
        "Charlie": ["charlie1.jpg"]
    }

    print("Building face database for known people...")
    for name, image_paths in known_people.items():
        existing_images = [p for p in image_paths if os.path.exists(p)]
        if existing_images:
            face_db.add_faces_from_images(
                name=name,
                image_paths=existing_images,
                face_embeddings=search.face_embeddings,
                min_confidence=0.5
            )
            print(f"  ✓ Added {name} ({len(existing_images)} image(s))")
        else:
            print(f"  ⚠ No images found for {name}")

    if len(face_db) > 0:
        print(f"\n✓ Face database created with {len(face_db)} entries")
        print(f"  Known people: {', '.join(face_db.get_all_names())}")

        # Save database
        face_db.save("known_faces.pkl")

        # Search in the database
        query_image = "query_person.jpg"
        if os.path.exists(query_image):
            print(f"\nSearching for person in {query_image}...")

            query_emb = search.face_embeddings.extract_largest_face_embedding(query_image)
            if query_emb is not None:
                matches = face_db.search(query_emb, threshold=0.6, top_k=3)

                if matches:
                    print("Matches:")
                    for match in matches:
                        print(f"  {match['name']}: {match['similarity']:.3f}")
                else:
                    print("  No matches found")
    else:
        print("⚠ No faces added to database (check image paths)")

    # Example 8: Batch processing
    print("\n" + "="*60)
    print("Example 8: Batch Face Detection")
    print("="*60)

    images_dir = "path/to/images/"

    if os.path.exists(images_dir):
        image_files = list(Path(images_dir).glob("*.jpg"))[:5]  # First 5 images

        if image_files:
            print(f"Processing {len(image_files)} images...")

            all_detections = search.face_embeddings.batch_detect_faces(
                [str(f) for f in image_files],
                min_confidence=0.5,
                show_progress=True
            )

            print("\nResults:")
            for img_file, detections in zip(image_files, all_detections):
                print(f"  {img_file.name}: {len(detections)} face(s)")
        else:
            print("No images found in directory")
    else:
        print(f"⚠ Directory not found: {images_dir}")

    print("\n" + "="*60)
    print("Face Search Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
