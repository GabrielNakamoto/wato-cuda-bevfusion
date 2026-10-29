#!/usr/bin/env python3
"""
Run TensorRT BEVFusion segmentation inference via libpybev across nuScenes samples
and render side-by-side composite frames (6 camera views + LiDAR BEV + 6-class BEV SegMap)
plus an MP4/AVI video.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# Ensure build/ is on sys.path so `import libpybev` resolves
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "build"))

import libpybev  # noqa: E402

CAM_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

MAP_CLASSES = [
    ("drivable_area", (166, 206, 227)),  # RGB
    ("ped_crossing", (31, 120, 180)),
    ("walkway", (178, 223, 138)),
    ("stop_line", (51, 160, 44)),
    ("carpark_area", (251, 154, 153)),
    ("divider", (227, 26, 28)),
]


def make_transform(rotation_quat, translation):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Quaternion(rotation_quat).rotation_matrix.astype(np.float32)
    T[:3, 3] = np.array(translation, dtype=np.float32)
    return T


def build_img_aug_matrix(orig_w=1600, orig_h=900, out_w=704, out_h=256, resize=0.48):
    # Matches ImageAug3D test_pipeline: resize=0.48 (768x432), crop=(32, 176, 736, 432) -> 704x256
    new_w = int(orig_w * resize)
    new_h = int(orig_h * resize)
    crop_w = int(new_w - out_w) // 2
    crop_h = int(new_h - out_h)
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = resize
    M[1, 1] = resize
    M[0, 3] = -crop_w
    M[1, 3] = -crop_h
    return M


def load_sample_inputs(nusc: NuScenes, sample: dict, dataroot: str):
    # LiDAR pose & points
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    lidar2ego = make_transform(lidar_cs["rotation"], lidar_cs["translation"])
    lidarego2global = make_transform(lidar_ep["rotation"], lidar_ep["translation"])
    lidar2global = lidarego2global @ lidar2ego
    global2lidar = np.linalg.inv(lidar2global).astype(np.float32)

    lidar_path = os.path.join(dataroot, lidar_sd["filename"])
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5).copy()
    pts[:, 4] = 0.0  # keyframe timestamp lag = 0
    points_fp16 = pts.astype(np.float16)

    imgs_rgb = []
    imgs_bgr = []
    camera2lidar_list = []
    intrinsics_list = []
    lidar2image_list = []
    aug_list = []
    aug_mat = build_img_aug_matrix()

    for cam in CAM_NAMES:
        cam_token = sample["data"][cam]
        cam_sd = nusc.get("sample_data", cam_token)
        cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_ep = nusc.get("ego_pose", cam_sd["ego_pose_token"])

        cam2ego = make_transform(cam_cs["rotation"], cam_cs["translation"])
        camego2global = make_transform(cam_ep["rotation"], cam_ep["translation"])
        cam2global = camego2global @ cam2ego

        cam2lidar = (global2lidar @ cam2global).astype(np.float32)
        lidar2cam = np.linalg.inv(cam2lidar).astype(np.float32)

        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = np.array(cam_cs.get("camera_intrinsics") or cam_cs.get("camera_intrinsic") or cam_cs.get("cam_intrinsic"), dtype=np.float32)

        lidar2img = (K @ lidar2cam).astype(np.float32)

        img_path = os.path.join(dataroot, cam_sd["filename"])
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        imgs_bgr.append(bgr)
        imgs_rgb.append(rgb)
        camera2lidar_list.append(cam2lidar)
        intrinsics_list.append(K)
        lidar2image_list.append(lidar2img)
        aug_list.append(aug_mat)

    images_np = np.stack(imgs_rgb, axis=0)[None, ...]  # (1, 6, 900, 1600, 3) uint8
    camera2lidar = np.stack(camera2lidar_list, axis=0)[None, ...]
    camera_intrinsics = np.stack(intrinsics_list, axis=0)[None, ...]
    lidar2image = np.stack(lidar2image_list, axis=0)[None, ...]
    img_aug_matrix = np.stack(aug_list, axis=0)[None, ...]

    return (
        images_np,
        imgs_bgr,
        pts,
        points_fp16,
        camera2lidar,
        camera_intrinsics,
        lidar2image,
        img_aug_matrix,
    )


def render_segmap_bgr(segmap: np.ndarray, size: int = 540, threshold: float = 0.5) -> np.ndarray:
    # segmap: (6, 200, 200) float32
    C, H, W = segmap.shape
    canvas = np.full((H, W, 3), 240, dtype=np.uint8)
    for c, (_, rgb) in enumerate(MAP_CLASSES[:C]):
        mask = segmap[c] >= threshold
        # Transpose + flip so ego vehicle faces UP
        mask_oriented = np.flipud(np.fliplr(mask.T))
        bgr = (rgb[2], rgb[1], rgb[0])
        canvas[mask_oriented] = bgr

    # Draw ego vehicle marker at center
    cy, cx = H // 2, W // 2
    cv2.rectangle(canvas, (cx - 2, cy - 4), (cx + 2, cy + 4), (20, 20, 20), -1)

    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)
    return resized


def render_lidar_bev_bgr(pts: np.ndarray, size: int = 540, max_range: float = 50.0) -> np.ndarray:
    canvas = np.full((size, size, 3), 20, dtype=np.uint8)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    keep = (np.abs(x) < max_range) & (np.abs(y) < max_range)
    x, y, z = x[keep], y[keep], z[keep]

    # Map x (right/left) and y (front/back) to top-down image coordinates matching segmap
    px = ((max_range - x) / (2 * max_range) * (size - 1)).astype(np.int32)
    py = ((max_range - y) / (2 * max_range) * (size - 1)).astype(np.int32)

    # Color by height z in [-3, 2]
    norm_z = np.clip((z + 3.0) / 5.0 * 255.0, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(norm_z.reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
    canvas[py, px] = colors

    c = size // 2
    cv2.rectangle(canvas, (c - 4, c - 8), (c + 4, c + 8), (0, 255, 255), -1)
    return canvas


def compose_frame(imgs_bgr, pts, segmap, scene_name, frame_idx) -> np.ndarray:
    # 6 cameras arranged in a 2x3 grid on the left:
    # Top row:    FRONT_LEFT (2), FRONT (0), FRONT_RIGHT (1)
    # Bottom row: BACK_LEFT (4),  BACK (3),  BACK_RIGHT (5)
    cam_w, cam_h = 400, 225
    order = [
        (2, "FRONT_LEFT", False),
        (0, "FRONT", False),
        (1, "FRONT_RIGHT", False),
        (4, "BACK_LEFT", True),
        (3, "BACK", True),
        (5, "BACK_RIGHT", True),
    ]
    tiles = []
    for idx, label, flip in order:
        im = cv2.resize(imgs_bgr[idx], (cam_w, cam_h), interpolation=cv2.INTER_AREA)
        if flip:
            im = cv2.flip(im, 1)
        cv2.putText(im, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)

    row_top = np.hstack(tiles[:3])   # 1200 x 225
    row_bot = np.hstack(tiles[3:])   # 1200 x 225
    cam_grid = np.vstack([row_top, row_bot])  # 1200 x 450

    # Legend bar below cameras (1200 x 90)
    legend = np.full((90, 1200, 3), 35, dtype=np.uint8)
    cv2.putText(
        legend,
        f"CUDA-BEVFusion TensorRT FP16 | {scene_name} | Frame {frame_idx:03d}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    lx = 15
    for name, rgb in MAP_CLASSES:
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(legend, (lx, 50), (lx + 24, 74), bgr, -1)
        cv2.rectangle(legend, (lx, 50), (lx + 24, 74), (200, 200, 200), 1)
        cv2.putText(legend, name, (lx + 32, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        lx += 190

    left_pane = np.vstack([cam_grid, legend])  # 1200 x 540

    # Right panes: LiDAR BEV (540x540) + SegMap BEV (540x540)
    lidar_bev = render_lidar_bev_bgr(pts, size=540)
    cv2.putText(lidar_bev, "LiDAR BEV", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    seg_bev = render_segmap_bgr(segmap, size=540)
    cv2.putText(seg_bev, "BEV Map Segmentation (Pred)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)

    return np.hstack([left_pane, lidar_bev, seg_bev])  # 2280 x 540


def main():
    parser = argparse.ArgumentParser(description="Render CUDA-BEVFusion segmentation video from nuScenes")
    parser.add_argument("--dataroot", default="/dataset", help="Path to nuScenes dataset")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version directory")
    parser.add_argument("--model", default="seg", help="Model directory under model/")
    parser.add_argument("--precision", default="fp16", help="TensorRT precision (fp16/int8)")
    parser.add_argument("--max-frames", type=int, default=80, help="Maximum number of frames to render")
    parser.add_argument("--fps", type=int, default=6, help="Output video frame rate")
    parser.add_argument("--out-dir", default="build/vis_frames", help="Directory to save individual frames and video")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading TensorRT BEVFusion ({args.model}, {args.precision})...")
    core = libpybev.load_bevfusion(
        f"model/{args.model}/build/camera.backbone.plan",
        f"model/{args.model}/build/camera.vtransform.plan",
        f"model/{args.model}/lidar.backbone.xyz.onnx",
        f"model/{args.model}/build/fuser.plan",
        f"model/{args.model}/build/head.seg.plan",
        args.precision,
    )
    core.print()

    print(f"Loading NuScenes ({args.version}) from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    video_path = os.path.join(args.out_dir, "bevfusion_seg.mp4")
    writer = None
    frame_count = 0

    for scene in nusc.scene:
        scene_name = scene["name"]
        sample_token = scene["first_sample_token"]
        while sample_token and frame_count < args.max_frames:
            sample = nusc.get("sample", sample_token)
            (
                images_np,
                imgs_bgr,
                pts,
                points_fp16,
                camera2lidar,
                camera_intrinsics,
                lidar2image,
                img_aug_matrix,
            ) = load_sample_inputs(nusc, sample, args.dataroot)

            core.update(np.ascontiguousarray(camera2lidar, dtype=np.float32), np.ascontiguousarray(camera_intrinsics, dtype=np.float32), np.ascontiguousarray(lidar2image, dtype=np.float32), np.ascontiguousarray(img_aug_matrix, dtype=np.float32))
            segmap = core.forward(images_np, points_fp16, with_normalization=True, with_dlpack=False)

            frame_bgr = compose_frame(imgs_bgr, pts, segmap, scene_name, frame_count)
            frame_path = os.path.join(args.out_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame_bgr)

            if writer is None:
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))

            writer.write(frame_bgr)
            frame_count += 1
            if frame_count % 10 == 0 or frame_count == args.max_frames:
                print(f"  Rendered {frame_count}/{args.max_frames} frames ({scene_name})")

            sample_token = sample["next"]
        if frame_count >= args.max_frames:
            break

    if writer is not None:
        writer.release()
    print(f"Saved {frame_count} frames to {args.out_dir}/ and video to {video_path}")


if __name__ == "__main__":
    main()
