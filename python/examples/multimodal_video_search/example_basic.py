"""
Basic Example: Video Search with CLIP + DINOv3

This example demonstrates:
1. Indexing a video
2. Searching with text queries
3. Searching with image queries
4. Hybrid text+image search
"""

from video_search import HybridVideoSearch
import os


def main():
    # Initialize the search system
    print("Initializing Hybrid Video Search...")
    search = HybridVideoSearch(
        db_path="./video_search_demo.db",
        device="cuda"  # Use "cpu" if no GPU available
    )

    # Example 1: Index a video
    print("\n" + "="*60)
    print("Example 1: Indexing a Video")
    print("="*60)

    video_path = "path/to/your/video.mp4"

    # Check if video exists (for demo, we'll skip if not)
    if os.path.exists(video_path):
        num_frames = search.index_video(
            video_path=video_path,
            video_id="demo_video_001",
            fps=2,  # Extract 2 frames per second
            max_frames=100  # Limit to 100 frames for demo
        )
        print(f"✓ Indexed {num_frames} frames")
    else:
        print(f"⚠ Video not found at {video_path}")
        print("Please provide a valid video path to test indexing")

    # Example 2: Text search
    print("\n" + "="*60)
    print("Example 2: Text Search")
    print("="*60)

    text_queries = [
        "person wearing red shirt",
        "American flag in the background",
        "outdoor scene with trees",
        "close-up of face"
    ]

    for query in text_queries:
        print(f"\nSearching for: '{query}'")
        try:
            results = search.search_text(query, limit=5)

            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Video: {result['video_id']}, "
                          f"Frame: {result['frame_id']}, "
                          f"Time: {result['timestamp']:.2f}s, "
                          f"Score: {result['score']:.3f}")
                    print(f"     Path: {result['frame_path']}")
            else:
                print("  No results found (database might be empty)")
        except Exception as e:
            print(f"  Error: {e}")
            print("  (Database might be empty - index a video first)")

    # Example 3: Image search
    print("\n" + "="*60)
    print("Example 3: Image Search")
    print("="*60)

    query_image = "path/to/query_image.jpg"

    if os.path.exists(query_image):
        print(f"Searching for frames similar to: {query_image}")
        results = search.search_image(
            image_path=query_image,
            limit=5,
            clip_weight=0.5,  # Weight for semantic similarity
            dino_weight=0.5   # Weight for visual similarity
        )

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Video: {result['video_id']}, "
                  f"Frame: {result['frame_id']}, "
                  f"Time: {result['timestamp']:.2f}s, "
                  f"Score: {result['score']:.3f}")
    else:
        print(f"⚠ Query image not found at {query_image}")
        print("Skipping image search example")

    # Example 4: Hybrid search (text + image)
    print("\n" + "="*60)
    print("Example 4: Hybrid Search (Text + Image)")
    print("="*60)

    if os.path.exists(query_image):
        print("Combining text and image queries...")
        results = search.hybrid_search(
            text="person wearing red",
            image=query_image,
            text_weight=0.6,    # Higher weight for text
            image_weight=0.4,   # Lower weight for image
            limit=5
        )

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Video: {result['video_id']}, "
                  f"Frame: {result['frame_id']}, "
                  f"Time: {result['timestamp']:.2f}s, "
                  f"Score: {result['score']:.3f}")
    else:
        print("Skipping hybrid search (query image not found)")

    # Example 5: Database statistics
    print("\n" + "="*60)
    print("Example 5: Database Statistics")
    print("="*60)

    stats = search.get_stats()
    print(f"Database Path: {search.db_path}")
    print(f"Tables: {stats['tables']}")
    print(f"Total Frames: {stats['total_frames']}")
    print(f"Unique Videos: {len(stats['videos'])}")
    if stats['videos']:
        print(f"Video IDs: {', '.join(stats['videos'])}")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
