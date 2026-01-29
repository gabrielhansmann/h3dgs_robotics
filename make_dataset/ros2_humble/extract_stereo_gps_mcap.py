import bisect
import numpy as np
import yaml
import os
import cv2
import argparse
from collections import defaultdict
from tqdm import tqdm
import pymap3d as pm
import datetime

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Imu, NavSatFix
from scipy.spatial.transform import Rotation as R, Slerp

# Configuration
BAG_PATH = "rosbag2_2026_01_17-14_14_33"
OUTPUT_BASE = "exported_data"

TARGET_FPS = 1.0
ENU = False
ORIGIN = None

GPS_TOPIC = "/fix"
CAM_INFO_TOPICS = [
    "/zed_multi/zed_front/right/camera_info", "/zed_multi/zed_front/left/camera_info",
    "/zed_multi/zed_rear/right/camera_info", "/zed_multi/zed_rear/left/camera_info",
    "/zed_multi/zed_right/right/camera_info", "/zed_multi/zed_right/left/camera_info",
    "/zed_multi/zed_left/right/camera_info", "/zed_multi/zed_left/left/camera_info"
]
IMU_TOPICS = [
    "/zed_multi/zed_front/imu/data", "/zed_multi/zed_rear/imu/data",
    "/zed_multi/zed_right/imu/data", "/zed_multi/zed_left/imu/data",
]
CAM_TOPICS = [
    "/zed_multi/zed_front/right/image_rect_color/compressed", "/zed_multi/zed_front/left/image_rect_color/compressed",
    "/zed_multi/zed_rear/right/image_rect_color/compressed", "/zed_multi/zed_rear/left/image_rect_color/compressed",
    "/zed_multi/zed_right/right/image_rect_color/compressed", "/zed_multi/zed_right/left/image_rect_color/compressed",
    "/zed_multi/zed_left/right/image_rect_color/compressed", "/zed_multi/zed_left/left/image_rect_color/compressed",
]

ALL_TOPICS = CAM_TOPICS + CAM_INFO_TOPICS + [GPS_TOPIC] + IMU_TOPICS

def lerp(v0, v1, alpha):
    return v0 + alpha * (v1 - v0)

def get_reader(path):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    return reader

def get_clean_cam_name(topic):
    """Converts /zed_multi/zed_front/left/... to zed_multi_zed_front_left"""
    clean = topic.replace("/image_rect_color/compressed", "").replace("/camera_info", "").strip('/')
    return clean.replace('/', '_')

def get_data_path(topic, base_dir):
    """Maps a topic to the specified directory structure."""
    if topic in CAM_TOPICS:
        return os.path.join(base_dir, "inputs", "images", get_clean_cam_name(topic))
    if topic in IMU_TOPICS:
        imu_name = topic.replace("/data", "").strip('/').replace('/', '_')
        return os.path.join(base_dir, "inputs", "imu", imu_name)
    if topic == GPS_TOPIC:
        return os.path.join(base_dir, "inputs", "robotics_gps")
    return os.path.join(base_dir, "inputs", "other", topic.strip('/').replace('/', '_'))

def extract_camera_configs(bag_path, cam_info_topics, config_output_path):
    print("\n--- Extracting Camera Calibration ---")
    reader = get_reader(bag_path)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    remaining_topics = set(cam_info_topics)
    config_data = {"cameras": {}}

    while reader.has_next() and remaining_topics:
        topic, data, _ = reader.read_next()
        if topic in remaining_topics:
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            yaml_key = get_clean_cam_name(topic)
            config_data["cameras"][yaml_key] = {
                "name": yaml_key.split('_')[-1],
                "resolution": {"width": int(msg.width), "height": int(msg.height)},
                "focal_length": {"fx": float(msg.k[0]), "fy": float(msg.k[4])},
                "optical_center": {"cx": float(msg.k[2]), "cy": float(msg.k[5])}
            }
            remaining_topics.remove(topic)

    with open(config_output_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    print(f"Done: {config_output_path} generated.")

def offset_feedback(schedule):
    """Prints diagnostic information about timing alignment."""
    topic_jitter_ms = defaultdict(list)
    for target_time, needed_timestamps in schedule:
        for top, ts_info in needed_timestamps.items():
            actual_ts = ts_info['needed'][0]
            offset_ms = abs(actual_ts - target_time) / 1e6
            topic_jitter_ms[top].append(offset_ms)

    print("\n" + "="*60)
    print(f"{'DATA GROUP':<15} | {'AVG JITTER (ms)':<18} | {'MAX JITTER (ms)':<15}")
    print("-" * 60)

    def print_group_stats(name, topics):
        group_offsets = [val for t in topics if t in topic_jitter_ms for val in topic_jitter_ms[t]]
        if group_offsets:
            avg_val, max_val = sum(group_offsets) / len(group_offsets), max(group_offsets)
            print(f"{name:<15} | {avg_val:>15.3f} ms | {max_val:>12.3f} ms")
        else:
            print(f"{name:<15} | {'No Data':>19} | {'N/A':>15}")

    print_group_stats("CAMERAS", CAM_TOPICS)
    print_group_stats("IMUs", IMU_TOPICS)
    print_group_stats("GPS", [GPS_TOPIC])
    print("="*60 + "\n")

# --- Interpolation Helpers ---

def interpolate_imu(m0, m1, alpha):
    out = Imu()
    t0_ns = m0.header.stamp.sec * 1e9 + m0.header.stamp.nanosec
    t1_ns = m1.header.stamp.sec * 1e9 + m1.header.stamp.nanosec
    target_ns = int(lerp(t0_ns, t1_ns, alpha))
    out.header.frame_id, out.header.stamp.sec, out.header.stamp.nanosec = m0.header.frame_id, int(target_ns // 1e9), int(target_ns % 1e9)
    for field in ['linear_acceleration', 'angular_velocity']:
        for axis in ['x', 'y', 'z']:
            setattr(getattr(out, field), axis, lerp(getattr(getattr(m0, field), axis), getattr(getattr(m1, field), axis), alpha))
    rot = Slerp([0, 1], R.from_quat([[m0.orientation.x, m0.orientation.y, m0.orientation.z, m0.orientation.w], 
                                     [m1.orientation.x, m1.orientation.y, m1.orientation.z, m1.orientation.w]]))
    q_out = rot([alpha])[0].as_quat()
    out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w = q_out
    return out

def interpolate_gps(m0, m1, alpha):
    out = NavSatFix()
    t0_ns = m0.header.stamp.sec * 1e9 + m0.header.stamp.nanosec
    t1_ns = m1.header.stamp.sec * 1e9 + m1.header.stamp.nanosec
    target_ns = int(lerp(t0_ns, t1_ns, alpha))
    out.header.stamp.sec, out.header.stamp.nanosec = int(target_ns // 1e9), int(target_ns % 1e9)
    out.latitude, out.longitude, out.altitude = lerp(m0.latitude, m1.latitude, alpha), lerp(m0.longitude, m1.longitude, alpha), lerp(m0.altitude, m1.altitude, alpha)
    return out

def imu_to_dict(msg):
    return {
        'header': {'stamp': {'sec': int(msg.header.stamp.sec), 'nanosec': int(msg.header.stamp.nanosec)}, 'frame_id': msg.header.frame_id},
        'orientation': {ax: float(getattr(msg.orientation, ax)) for ax in 'xyzw'},
        'angular_velocity': {ax: float(getattr(msg.angular_velocity, ax)) for ax in 'xyz'},
        'linear_acceleration': {ax: float(getattr(msg.linear_acceleration, ax)) for ax in 'xyz'}
    }

def gps_to_dict(msg):
    global ORIGIN
    if not ORIGIN: ORIGIN = (msg.latitude, msg.longitude, msg.altitude)    
    if ENU: coords = pm.geodetic2enu(msg.latitude, msg.longitude, msg.altitude, ORIGIN[0], ORIGIN[1], ORIGIN[2])
    else: coords = (msg.latitude, msg.longitude, msg.altitude)

    return {
        'header': {'stamp': {'sec': int(msg.header.stamp.sec), 'nanosec': int(msg.header.stamp.nanosec)}, 'frame_id': msg.header.frame_id},
        'x': float(coords[0]), 
        'y': float(coords[1]), 
        'z': float(coords[2])
    }

def clean_empty_dirs(path):
    """Recursively deletes empty directories."""
    for root, dirs, files in os.walk(path, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

def synchronize():
    bag_name = os.path.basename(BAG_PATH.rstrip('/'))
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_output_dir = os.path.join(OUTPUT_BASE, bag_name, timestamp_str)
    os.makedirs(bag_output_dir, exist_ok=True)
    
    # print(f'hz: {TARGET_FPS}, enu: {ENU}')
    
    extract_camera_configs(BAG_PATH, CAM_INFO_TOPICS, os.path.join(bag_output_dir, "robotics_config.yaml"))
    
    print("Indexing timestamps...")
    reader = get_reader(BAG_PATH)
    topic_indices = {topic: [] for topic in ALL_TOPICS}
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, _, timestamp = reader.read_next()
        if topic in topic_indices: topic_indices[topic].append(timestamp)
    del reader

    active_topics = {t: indices for t, indices in topic_indices.items() if indices}
    if not active_topics: return
    
    start_time, end_time = max(ts[0] for ts in active_topics.values()), min(ts[-1] for ts in active_topics.values())
    interval_ns, current_target, schedule = int(1e9 / TARGET_FPS), start_time, []

    while current_target <= end_time:
        slot = {}
        for topic, ts_list in active_topics.items():
            idx = bisect.bisect_left(ts_list, current_target)
            if topic in IMU_TOPICS or topic == GPS_TOPIC:
                if 0 < idx < len(ts_list):
                    t0, t1 = ts_list[idx-1], ts_list[idx]
                    slot[topic] = {'needed': [t0, t1], 'alpha': (current_target - t0) / (t1 - t0), 'mode': 'lerp'}
                else:
                    slot[topic] = {'needed': [ts_list[idx if idx < len(ts_list) else -1]], 'mode': 'closest'}
            else:
                best_ts = ts_list[idx] if (idx < len(ts_list) and (idx == 0 or abs(ts_list[idx]-current_target) < abs(ts_list[idx-1]-current_target))) else ts_list[idx-1]
                slot[topic] = {'needed': [best_ts], 'mode': 'closest'}
        schedule.append((current_target, slot))
        current_target += interval_ns

    offset_feedback(schedule)

    for topic in ALL_TOPICS:
        os.makedirs(get_data_path(topic, bag_output_dir), exist_ok=True)

    lookup = defaultdict(list)
    for slot_idx, (_, needed_dict) in enumerate(schedule):
        for topic, info in needed_dict.items():
            for ts in info['needed']:
                lookup[ts].append((slot_idx, topic, info.get('mode'), info.get('alpha')))

    lerp_buffer = {}
    data_reader = get_reader(BAG_PATH)
    pbar = tqdm(total=len(schedule), desc="Exporting Bundles")
    processed_slots = set()

    while data_reader.has_next():
        topic, data, timestamp = data_reader.read_next()
        if timestamp in lookup:
            for slot_idx, target_topic, mode, alpha in lookup[timestamp]:
                if target_topic != topic: continue
                msg = deserialize_message(data, get_message(topic_types[topic]))
                save_path = os.path.join(get_data_path(topic, bag_output_dir), f"{slot_idx+1:04d}")

                if topic in CAM_TOPICS:
                    cv2.imwrite(f"{save_path}.jpg", cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR))
                elif topic in IMU_TOPICS or topic == GPS_TOPIC:
                    if mode == 'lerp':
                        key = (slot_idx, topic)
                        if key not in lerp_buffer: lerp_buffer[key] = msg
                        else:
                            m0, m1 = lerp_buffer.pop(key), msg
                            interp = interpolate_gps(m0, m1, alpha) if topic == GPS_TOPIC else interpolate_imu(m0, m1, alpha)
                            with open(f"{save_path}.yaml", 'w') as f:
                                yaml.dump(gps_to_dict(interp) if topic == GPS_TOPIC else imu_to_dict(interp), f, sort_keys=False)
                    else:
                        with open(f"{save_path}.yaml", 'w') as f:
                            yaml.dump(gps_to_dict(msg) if topic == GPS_TOPIC else imu_to_dict(msg), f, sort_keys=False)
                
                if slot_idx not in processed_slots:
                    processed_slots.add(slot_idx)
                    pbar.update(1)
    pbar.close()
    
    print("Cleaning up empty directories...")
    clean_empty_dirs(bag_output_dir)
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rosbag', default=BAG_PATH, type=str, required=True, help='path to rosbag')
    parser.add_argument('--enu', default=ENU, type=bool, help='if enu or lat/long')
    parser.add_argument('--hz', default=TARGET_FPS, type=float, help='target hz to synchronize')
    args = parser.parse_args()

    BAG_PATH = str(args.rosbag)
    ENU = bool(args.enu)
    TARGET_FPS = float(args.hz)

    synchronize()