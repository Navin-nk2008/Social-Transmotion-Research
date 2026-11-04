import os
import json
import glob
import sys

def audit_directory(base_dir, dataset_name):
    print(f"\n==================================================")
    print(f"Auditing Dataset: {dataset_name} at '{base_dir}'")
    print(f"==================================================")
    splits = ['train', 'val', 'test']
    total_files = 0
    total_scenes = 0
    total_tracks = 0

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[ERROR] Split directory missing: {split_dir}")
            continue

        ndjson_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.ndjson')])
        print(f"\n--- Split: {split} ---")
        print(f"Path: {split_dir}")
        print(f"NDJSON file count: {len(ndjson_files)}")
        total_files += len(ndjson_files)

        split_scenes = 0
        split_tracks = 0
        missing_features = 0
        nan_in_tracks = 0
        read_errors = 0
        sample_keys = None

        for fname in ndjson_files:
            fpath = os.path.join(split_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line_idx, line in enumerate(f):
                        data = json.loads(line)
                        if 'scene' in data:
                            split_scenes += 1
                        elif 'track' in data:
                            split_tracks += 1
                            track = data['track']
                            if sample_keys is None:
                                sample_keys = list(track.keys())

                            # Check expected fields
                            if dataset_name == 'jta_all_visual_cues':
                                # Check trajectory, 3D BB, 2D BB, 3D pose, 2D pose
                                expected_fields = ['f', 'p', 'x', 'y', 'h', 'w', 'l', 'rot_z',
                                                   'bb_left', 'bb_top', 'bb_width', 'bb_height',
                                                   'x0', 'y0', 'z0', 'xx0', 'yy0']
                                for ef in expected_fields:
                                    if ef not in track:
                                        missing_features += 1
                                        break
                            elif dataset_name == 'jrdb_2dbox':
                                expected_fields = ['f', 'p', 'x', 'y', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
                                for ef in expected_fields:
                                    if ef not in track:
                                        missing_features += 1
                                        break
                        else:
                            pass
            except Exception as e:
                read_errors += 1
                print(f"[ERROR] Failed reading {fpath}: {e}")

        print(f"Readable files: {len(ndjson_files) - read_errors}/{len(ndjson_files)}")
        print(f"Total scenes: {split_scenes}")
        print(f"Total track frames: {split_tracks}")
        print(f"Missing expected features: {missing_features}")
        print(f"Sample track keys count: {len(sample_keys) if sample_keys else 0}")
        if sample_keys:
            print(f"Sample track keys (first 10): {sample_keys[:10]}...")

    return total_files

if __name__ == '__main__':
    audit_directory('data/jta_all_visual_cues', 'jta_all_visual_cues')
    audit_directory('data/jrdb_2dbox', 'jrdb_2dbox')

