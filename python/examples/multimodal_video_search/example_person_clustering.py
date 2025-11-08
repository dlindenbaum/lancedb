"""
Example: Person Clustering and Re-Identification

This example demonstrates:
1. Tracking people across video frames
2. Clustering tracked people using face recognition
3. Identifying unique individuals across tracks
4. Re-identifying people across multiple videos
5. Assigning names to clusters
6. Searching for specific people
"""

from video_search import HybridVideoSearch
import os
from pathlib import Path


def main():
    print("="*70)
    print("Person Clustering and Re-Identification Example")
    print("="*70)

    # Initialize with both tracking and face recognition
    print("\n1. Initializing video search with tracking + face recognition...")
    search = HybridVideoSearch(
        db_path="./person_clustering_demo.db",
        device="cuda",  # Use "cpu" if no GPU available
        use_people_tracking=True,  # Enable YOLO + ByteTrack
        use_face_detection=True,   # Enable face recognition for clustering
        face_model="Facenet512",   # Face recognition model
        yolo_conf_threshold=0.5,
        use_dino=False  # Disable DINOv3 for faster processing
    )

    # Example video paths
    video_paths = {
        "video_001": "path/to/video1.mp4",
        "video_002": "path/to/video2.mp4",
        "video_003": "path/to/video3.mp4"
    }

    # Check if videos exist
    existing_videos = {vid: path for vid, path in video_paths.items()
                      if os.path.exists(path)}

    if not existing_videos:
        print("\n⚠ No videos found. Please update the video_paths dictionary")
        print("For this example, we'll demonstrate the workflow conceptually.")
        print("\nTo test with real videos, update the paths in this script:")
        print("  video_paths = {")
        print("      'video_001': '/path/to/your/video1.mp4',")
        print("      'video_002': '/path/to/your/video2.mp4',")
        print("  }")
        return

    # Step 1: Track people in multiple videos
    print("\n2. Tracking people in videos...")
    print("-" * 70)

    all_tracks = []
    for video_id, video_path in existing_videos.items():
        print(f"\nProcessing {video_id}...")
        tracks = search.track_people_in_video(
            video_path=video_path,
            video_id=video_id,
            fps=2,  # 2 frames per second
            save_tracks=True
        )
        all_tracks.extend(tracks)
        print(f"  Found {len(tracks)} people in {video_id}")

    print(f"\n✓ Total tracks across all videos: {len(all_tracks)}")

    # Step 2: Cluster tracked people to identify unique individuals
    print("\n3. Clustering tracked people using face recognition...")
    print("-" * 70)

    clusters = search.cluster_tracked_people(
        video_ids=list(existing_videos.keys()),
        video_paths=existing_videos,
        similarity_threshold=0.6,  # Faces with >60% similarity = same person
        clustering_method="dbscan",  # or "agglomerative"
        save_clusters=True
    )

    if not clusters:
        print("⚠ No clusters created (no faces detected or clustering failed)")
        return

    print(f"\n✓ Identified {len(clusters)} unique people from {len(all_tracks)} tracks")

    # Step 3: Analyze clusters
    print("\n4. Cluster Analysis:")
    print("-" * 70)

    for cluster_id, cluster in sorted(clusters.items())[:10]:  # Show first 10
        print(f"\nPerson {cluster_id}:")
        print(f"  Tracks: {cluster.num_tracks()}")
        print(f"  Videos: {cluster.num_videos()}")
        print(f"  Confidence: {cluster.confidence:.3f}")
        print(f"  Appears in videos: {set(cluster.video_ids)}")

        # Show track details
        for track_id, video_id in zip(cluster.track_ids[:3], cluster.video_ids[:3]):
            appearances = search.get_person_timeline(track_id, video_id)
            if appearances:
                print(f"    - Track {track_id} in {video_id}: "
                      f"{appearances['duration']:.1f}s, "
                      f"{appearances['num_detections']} frames")

        if cluster.num_tracks() > 3:
            print(f"    ... and {cluster.num_tracks() - 3} more tracks")

    # Step 4: Get clustering statistics
    print("\n5. Clustering Statistics:")
    print("-" * 70)

    stats = search.get_clustering_statistics()
    print(f"Unique people: {stats['num_clusters']}")
    print(f"Total tracks: {stats['total_tracks']}")
    print(f"Avg tracks per person: {stats['avg_tracks_per_person']:.1f}")
    print(f"Max tracks for one person: {stats['max_tracks_per_person']}")
    print(f"Avg videos per person: {stats['avg_videos_per_person']:.1f}")
    print(f"Avg cluster confidence: {stats['avg_cluster_confidence']:.3f}")

    # Step 5: Assign names to people
    print("\n6. Assigning Names to People:")
    print("-" * 70)

    # Example: Manually assign names based on inspection
    # In practice, you'd identify people and assign names
    name_mapping = {
        0: "Person A",
        1: "Person B",
        2: "Person C",
        # Add more as needed
    }

    search.assign_person_names(name_mapping)
    print(f"✓ Assigned names to {len(name_mapping)} people")

    # Step 6: Search for a specific person by name
    print("\n7. Searching for Specific People:")
    print("-" * 70)

    for name in name_mapping.values():
        appearances = search.search_for_person(name)
        if appearances:
            print(f"\n{name}:")
            print(f"  Total appearances: {len(appearances)}")

            total_duration = sum(app['duration'] for app in appearances)
            print(f"  Total screen time: {total_duration:.1f}s")

            videos = set(app['video_id'] for app in appearances)
            print(f"  Appears in videos: {videos}")

    # Step 7: Identify a person from a photo
    print("\n8. Identifying Person from Photo:")
    print("-" * 70)

    query_image = "path/to/person_photo.jpg"

    if os.path.exists(query_image):
        result = search.identify_person_from_image(
            query_image=query_image,
            min_similarity=0.6
        )

        if result:
            print(f"✓ Match found!")
            print(f"  Cluster ID: {result['cluster_id']}")
            print(f"  Person: {result['person_name'] or 'Unknown'}")
            print(f"  Similarity: {result['similarity']:.3f}")
            print(f"  Appears in {result['num_tracks']} tracks")
            print(f"  Appears in {result['num_videos']} videos")
        else:
            print("⚠ No match found")
    else:
        print(f"⚠ Query image not found: {query_image}")
        print("To test person identification, provide a valid image path")

    # Step 8: Visualize a person's appearances
    print("\n9. Creating Visualizations:")
    print("-" * 70)

    if clusters:
        # Visualize first person
        first_cluster_id = list(clusters.keys())[0]

        viz_output = f"person_{first_cluster_id}_appearances.jpg"

        search.visualize_person_cluster(
            cluster_id=first_cluster_id,
            output_path=viz_output,
            video_paths=existing_videos,
            max_images=10
        )

        if os.path.exists(viz_output):
            print(f"✓ Saved visualization: {viz_output}")

    # Step 9: Find people appearing across multiple videos
    print("\n10. Cross-Video Analysis:")
    print("-" * 70)

    multi_video_people = []
    for cluster_id, cluster in clusters.items():
        if cluster.num_videos() > 1:
            multi_video_people.append({
                'cluster_id': cluster_id,
                'name': cluster.person_name,
                'num_videos': cluster.num_videos(),
                'num_tracks': cluster.num_tracks(),
                'videos': set(cluster.video_ids)
            })

    if multi_video_people:
        print(f"Found {len(multi_video_people)} people appearing in multiple videos:")
        for person in sorted(multi_video_people, key=lambda x: x['num_videos'], reverse=True):
            name = person['name'] or f"Person {person['cluster_id']}"
            print(f"\n{name}:")
            print(f"  Appears in {person['num_videos']} videos: {person['videos']}")
            print(f"  Total tracks: {person['num_tracks']}")
    else:
        print("No people found appearing in multiple videos")

    # Step 10: Get all appearances of a specific person
    print("\n11. Detailed Person Timeline:")
    print("-" * 70)

    if clusters:
        first_cluster_id = list(clusters.keys())[0]
        cluster = clusters[first_cluster_id]
        person_name = cluster.person_name or f"Person {first_cluster_id}"

        print(f"\nTimeline for {person_name}:")

        appearances = search.get_person_appearances(first_cluster_id)

        for app in appearances:
            print(f"\n  Video: {app['video_id']}")
            print(f"    Track ID: {app['track_id']}")
            print(f"    Duration: {app['duration']:.2f}s")
            print(f"    Frames: {app['first_frame']} to {app['last_frame']}")
            print(f"    Time: {app['first_timestamp']:.1f}s to {app['last_timestamp']:.1f}s")
            print(f"    Detections: {app['num_detections']}")

    # Step 11: Export cluster data
    print("\n12. Exporting Cluster Data:")
    print("-" * 70)

    if hasattr(search, 'person_clusterer') and search.person_clusterer:
        export_file = "person_clusters.json"
        search.person_clusterer.export_clusters(export_file)

        if os.path.exists(export_file):
            print(f"✓ Cluster data exported to: {export_file}")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"✓ Processed {len(existing_videos)} videos")
    print(f"✓ Tracked {len(all_tracks)} individual tracks")
    print(f"✓ Identified {len(clusters)} unique people")
    print(f"✓ {len(multi_video_people)} people appear across multiple videos")
    print(f"✓ Assigned names to {len(name_mapping)} people")

    print("\nYou can now:")
    print("  - Search for people by name: search_for_person('Person A')")
    print("  - Identify people from photos: identify_person_from_image('photo.jpg')")
    print("  - Get person timeline: get_person_appearances(cluster_id)")
    print("  - Visualize appearances: visualize_person_cluster(cluster_id, ...)")
    print("="*70)


if __name__ == "__main__":
    main()
