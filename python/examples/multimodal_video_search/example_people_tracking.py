"""
Example: People Tracking in Videos

This example demonstrates:
1. Tracking people across video frames using YOLO + ByteTrack
2. Storing track information in LanceDB
3. Retrieving and analyzing tracked people
4. Visualizing person tracks
5. Getting tracking statistics
"""

from video_search import HybridVideoSearch
import os
from pathlib import Path


def main():
    print("="*70)
    print("People Tracking Example")
    print("="*70)

    # Initialize the search system with people tracking enabled
    print("\n1. Initializing video search with people tracking...")
    search = HybridVideoSearch(
        db_path="./people_tracking_demo.db",
        device="cuda",  # Use "cpu" if no GPU available
        use_people_tracking=True,  # Enable YOLO + ByteTrack tracking
        yolo_conf_threshold=0.5,  # Confidence threshold for person detection
        use_dino=False  # Disable DINOv3 for faster processing
    )

    # Example video path
    video_path = "path/to/your/video.mp4"
    video_id = "demo_video_001"

    # Check if video exists
    if not os.path.exists(video_path):
        print(f"\n⚠ Video not found at {video_path}")
        print("Please provide a valid video path to test people tracking")
        print("\nTo test with a different video, update the 'video_path' variable")
        return

    # Track people in the video
    print(f"\n2. Tracking people in video: {Path(video_path).name}")
    print("This will detect and track all people throughout the video...")

    tracks = search.track_people_in_video(
        video_path=video_path,
        video_id=video_id,
        fps=2,  # Process 2 frames per second
        max_frames=None,  # Process all frames
        save_tracks=True  # Save to database
    )

    print(f"\n✓ Found {len(tracks)} unique people in the video")

    # Display track information
    if tracks:
        print("\n3. Track Details:")
        print("-" * 70)
        for i, track in enumerate(tracks[:10], 1):  # Show first 10
            print(f"\nPerson #{i} (Track ID: {track.track_id})")
            print(f"  First seen: {track.first_timestamp:.2f}s (frame {track.first_frame})")
            print(f"  Last seen:  {track.last_timestamp:.2f}s (frame {track.last_frame})")
            print(f"  Duration:   {track.duration():.2f}s")
            print(f"  Appearances: {track.num_detections()} frames")
            print(f"  Avg confidence: {sum(track.confidences) / len(track.confidences):.3f}")

        if len(tracks) > 10:
            print(f"\n... and {len(tracks) - 10} more people")

    # Get tracked people from database
    print("\n4. Querying tracked people from database...")

    # Get all tracks
    all_tracks = search.get_tracked_people(video_id=video_id)
    print(f"Total tracks in database: {len(all_tracks)}")

    # Get tracks with minimum duration
    long_tracks = search.get_tracked_people(
        video_id=video_id,
        min_duration=1.0,  # At least 1 second
        min_detections=3   # At least 3 detections
    )
    print(f"People appearing for >1s: {len(long_tracks)}")

    # Get detailed timeline for a specific person
    if tracks:
        print("\n5. Detailed Timeline for First Person:")
        print("-" * 70)

        first_track_id = tracks[0].track_id
        timeline = search.get_person_timeline(
            track_id=first_track_id,
            video_id=video_id
        )

        if timeline:
            print(f"Track ID: {timeline['track_id']}")
            print(f"Duration: {timeline['duration']:.2f}s")
            print(f"Appearances: {timeline['num_detections']} frames")
            print(f"\nTimestamp sequence:")
            for i, (ts, conf) in enumerate(zip(timeline['timestamps'][:5],
                                              timeline['confidences'][:5])):
                print(f"  {i+1}. {ts:6.2f}s - confidence: {conf:.3f}")
            if timeline['num_detections'] > 5:
                print(f"  ... and {timeline['num_detections'] - 5} more")

    # Create visualization for a track
    if tracks and os.path.exists(video_path):
        print("\n6. Creating track visualization...")

        first_track_id = tracks[0].track_id
        output_viz = f"track_{first_track_id}_visualization.jpg"

        search.visualize_person_track(
            track_id=first_track_id,
            video_id=video_id,
            video_path=video_path,
            output_path=output_viz,
            max_frames=10  # Show up to 10 frames
        )

        if os.path.exists(output_viz):
            print(f"✓ Saved visualization to: {output_viz}")

    # Get tracking statistics
    print("\n7. People Tracking Statistics:")
    print("-" * 70)

    stats = search.get_people_tracking_statistics(video_id=video_id)

    print(f"Total unique people: {stats['total_tracks']}")
    print(f"Average duration: {stats['avg_duration']:.2f}s")
    print(f"Longest appearance: {stats['max_duration']:.2f}s")
    print(f"Shortest appearance: {stats['min_duration']:.2f}s")
    print(f"Average detections per person: {stats['avg_detections']:.1f}")
    print(f"Max detections: {stats['max_detections']}")
    print(f"Min detections: {stats['min_detections']}")

    # Advanced: Filter tracks by criteria
    print("\n8. Advanced Filtering:")
    print("-" * 70)

    # Find people who appear for a long time
    prominent_people = search.get_tracked_people(
        video_id=video_id,
        min_duration=5.0,  # At least 5 seconds
        min_detections=10  # At least 10 frames
    )
    print(f"Prominent people (>5s, >10 frames): {len(prominent_people)}")

    # Find brief appearances
    brief_appearances = search.get_tracked_people(
        video_id=video_id,
        min_duration=0.0,
        min_detections=1
    )
    brief_only = [t for t in brief_appearances if t['duration'] < 1.0]
    print(f"Brief appearances (<1s): {len(brief_only)}")

    # Export track data
    print("\n9. Exporting track data...")
    if tracks:
        from people_tracking import PeopleTracker

        # Create a temporary tracker just for export
        tracker = PeopleTracker(device="cpu")
        tracker.tracks = {t.track_id: t for t in tracks}

        export_file = "people_tracks.json"
        tracker.export_tracks(export_file)

        if os.path.exists(export_file):
            print(f"✓ Track data exported to: {export_file}")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"✓ Processed video: {video_id}")
    print(f"✓ Found {len(tracks)} unique people")
    print(f"✓ Tracks saved to database: {search.db_path}")
    print("\nYou can now:")
    print("  - Query tracks using get_tracked_people()")
    print("  - Get detailed timelines with get_person_timeline()")
    print("  - Visualize tracks with visualize_person_track()")
    print("  - Get statistics with get_people_tracking_statistics()")
    print("="*70)


if __name__ == "__main__":
    main()
