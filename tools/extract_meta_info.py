import argparse
import json
import os

# -----
# [{'vid': , 'kps': , 'other':},
#  {'vid': , 'kps': , 'other':}]
# -----
# python tools/extract_meta_info.py --root_path /path/to/video_dir --dataset_name fashion
# -----
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--meta_info_name", type=str)
parser.add_argument("--pose_suffix", type=str, default="_dwpose")
parser.add_argument("--depth_suffix", type=str, default=None)
parser.add_argument("--seg_suffix", type=str, default=None)

args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name

pose_dir = args.root_path + args.pose_suffix
depth_dir = None
if args.depth_suffix is not None:
    depth_dir = args.root_path + args.depth_suffix
seg_dir = None
if args.seg_suffix is not None:
    seg_dir = args.root_path + args.seg_suffix

# collect all video_folder paths
video_mp4_paths = set()
for root, dirs, files in os.walk(args.root_path):
    for name in files:
        if name.endswith(".mp4"):
            video_mp4_paths.add(os.path.join(root, name))
video_mp4_paths = list(video_mp4_paths)

meta_infos = []
for video_mp4_path in video_mp4_paths:
    relative_video_name = os.path.relpath(video_mp4_path, args.root_path)
    kps_path = os.path.join(pose_dir, relative_video_name)
    if not os.path.exists(kps_path):
        continue
    if depth_dir is not None:
        depth_path = os.path.join(depth_dir, relative_video_name)
        if not os.path.exists(depth_path):
            continue
    if seg_dir is not None:
        seg_path = os.path.join(seg_dir, relative_video_name)
        if not os.path.exists(seg_path):
            continue
    meta_info = {"video_path": video_mp4_path, "kps_path": kps_path}
    if depth_dir is not None:
        meta_info["depth_path"] = depth_path
    if seg_dir is not None:
        meta_info["seg_path"] = seg_path
    meta_infos.append(meta_info)

print(f"Number of videos: {len(meta_infos)}")
json.dump(meta_infos, open(f"./data/{args.meta_info_name}_meta.json", "w"))

