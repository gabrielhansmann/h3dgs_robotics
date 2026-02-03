import bisect
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import cv2
import argparse
import datetime
from collections import defaultdict
from tqdm import tqdm
import json
import pymap3d as pm

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Imu, NavSatFix
from scipy.spatial.transform import Rotation as R, Slerp

# --- Configuration & Globals ---
ORIGIN = None

def lerp(v0, v1, alpha):
    return v0 + alpha * (v1 - v0)

def get_reader(path, topics=None):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr', 
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)
    
    if topics:
        filt = rosbag2_py.StorageFilter(topics=topics)
        reader.set_filter(filt)
        
    return reader

def get_clean_cam_name(topic):
    clean = topic.replace("/image_rect_color/compressed", "").replace("/camera_info", "").strip('/')
    return clean.replace('/', '_')

def get_data_path(topic, base_dir, cam_topics, imu_topic, gps_topic):
    if topic in cam_topics:
        return os.path.join(base_dir, "inputs", "images", get_clean_cam_name(topic))
    if topic == imu_topic:
        # imu_name = topic.replace("/data", "").strip('/').replace('/', '_')
        return os.path.join(base_dir, "inputs", "robotics_imu")
    if topic == gps_topic:
        return os.path.join(base_dir, "inputs", "robotics_gps")
    return os.path.join(base_dir, "inputs", "other", topic.strip('/').replace('/', '_'))

def extract_camera_configs(bag_path, cam_info_topics, config_output_path):
    """Saves camera intrinsic parameters to a YAML file."""
    print("\n--- Extracting Camera Calibration ---")
    reader = get_reader(bag_path, topics=cam_info_topics)
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

def extract_rig_config(cam_topics, config_output_path):
    """
    Generates a robotics_rig_config.json matching the undistorted image prefixes.
    """
    print("\n--- Generating Rig Configuration ---")
    rig_cameras = []
    
    for i, topic in enumerate(cam_topics):
        image_prefix = get_clean_cam_name(topic)
        cam_entry = {
            "image_prefix": image_prefix
        }
        
        # Set the first camera as the reference sensor (anchor of the rig)
        if i == 0:
            cam_entry["ref_sensor"] = True
        else:
            cam_entry["cam_from_rig_rotation"] = [0, 0, 0, 0]
            cam_entry["cam_from_rig_translation"] = [0, 0, 0]

        rig_cameras.append(cam_entry)

    rig_data = [
        {
            "cameras": rig_cameras
        }
    ]

    with open(config_output_path, 'w') as f:
        json.dump(rig_data, f, indent=4)
        
    print(f"Done: {config_output_path} generated.")

# --- Diagnostic Visualization ---

def offset_feedback(schedule, cam_topics, imu_topic, gps_topic, output_dir):
    """Prints separate diagnostic tables and individual plots for each sensor group."""
    topic_jitter_ms = defaultdict(list)
    
    for target_time, needed_timestamps in schedule:
        for top, ts_info in needed_timestamps.items():
            actual_ts = ts_info.get('diag_ts', ts_info['needed'][0])
            offset_ms = (actual_ts - target_time) / 1e6
            topic_jitter_ms[top].append(offset_ms)

    groups = [
        ("CAMERAS", cam_topics),
        ("IMU", [imu_topic]),
        ("GPS", [gps_topic])
    ]

    for group_name, topics in groups:
        active_in_group = [t for t in topics if t in topic_jitter_ms]
        if not active_in_group: 
            print(f"topic {topic} not found. Continuing...")
            continue
            
        print("\n" + "="*85)
        print(f" {group_name} DIAGNOSTICS")
        print("-" * 85)
        print(f"{'TOPIC':<60} | {'MEAN (ms)':<10} | {'STD (σ)':<8}")
        print("-" * 85)
        
        group_data = []
        for topic in sorted(active_in_group):
            data = topic_jitter_ms[topic]
            group_data.extend(data)
            mu, sigma = np.mean(data), np.std(data)
            print(f"{topic:<60} | {mu:>9.3f} | {sigma:>8.3f}")
        
        g_mu, g_sigma = np.mean(group_data), np.std(group_data)
        print("-" * 85)
        print(f"{'GROUP AGGREGATE':<60} | {g_mu:>9.3f} | {g_sigma:>8.3f}")
        print("="*85)

        if os.environ.get('DISPLAY'):
            try:
                plt.figure(figsize=(10, 5))
                plt.hist(group_data, bins=40, density=True, alpha=0.3, color='skyblue', 
                         label=f"{group_name} Dist (N={len(group_data)})")
                
                if g_sigma > 0:
                    x = np.linspace(min(group_data)-10, max(group_data)+10, 300)
                    y = 1/(g_sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - g_mu)**2 / (2 * g_sigma**2) )
                    plt.plot(x, y, color='blue', linewidth=2, label=f'Gaussian Fit (σ={g_sigma:.2f})')

                max_dev = max(np.max(np.abs(group_data)), 20)
                plt.xlim(-max_dev * 1.1, max_dev * 1.1)
                plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Target (0ms)')
                plt.axvline(g_mu, color='green', linestyle=':', label=f'Mean ({g_mu:.1f}ms)')
                plt.title(f"{group_name} Synchronization Jitter & Latency")
                plt.xlabel("Offset from Target (ms)")
                plt.ylabel("Probability Density")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                file_name = f"jitter_{group_name.lower()}.png"
                plt.savefig(os.path.join(output_dir, file_name), dpi=300)
                print(f"Saved diagnostic plot: {file_name}")

                plt.show()
            except Exception as e:
                print(f"Plot failed for {group_name}: {e}")

# --- Interpolation & Transformation ---

def interpolate_imu(m0, m1, alpha):
    out = Imu()
    t0_ns = m0.header.stamp.sec * 1e9 + m0.header.stamp.nanosec
    t1_ns = m1.header.stamp.sec * 1e9 + m1.header.stamp.nanosec
    target_ns = int(lerp(t0_ns, t1_ns, alpha))
    out.header.frame_id = m0.header.frame_id
    out.header.stamp.sec, out.header.stamp.nanosec = int(target_ns // 1e9), int(target_ns % 1e9)
    for field in ['linear_acceleration', 'angular_velocity']:
        for axis in ['x', 'y', 'z']:
            v0, v1 = getattr(getattr(m0, field), axis), getattr(getattr(m1, field), axis)
            setattr(getattr(out, field), axis, lerp(v0, v1, alpha))
    rots = R.from_quat([[m0.orientation.x, m0.orientation.y, m0.orientation.z, m0.orientation.w], 
                        [m1.orientation.x, m1.orientation.y, m1.orientation.z, m1.orientation.w]])
    q_out = Slerp([0, 1], rots)([alpha])[0].as_quat()
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

def gps_to_dict(msg):
    global ORIGIN    
    if not ORIGIN: ORIGIN = (msg.latitude, msg.longitude, msg.altitude)
    
    coords_latlon = (msg.latitude, msg.longitude, msg.altitude)
    coords_enu = pm.geodetic2enu(msg.latitude, msg.longitude, msg.altitude, ORIGIN[0], ORIGIN[1], ORIGIN[2])
    
    return {
        'header': {'stamp': {'sec': int(msg.header.stamp.sec), 'nanosec': int(msg.header.stamp.nanosec)}, 'frame_id': msg.header.frame_id},
        'latitude': float(coords_latlon[0]), 
        'longitude': float(coords_latlon[1]), 
        'altitude': float(coords_latlon[2]),
        'x': float(coords_enu[0]), 
        'y': float(coords_enu[1]), 
        'z': float(coords_enu[2])
    }

def imu_to_dict(msg):
    # COLMAP X = Robot Y (Left) * −1 = Robot Right
    # COLMAP Y = Robot Z (Up) * −1 = Robot Down
    # COLMAP Z = Robot X (Forward)
    q_ros = [float(getattr(msg.orientation, ax)) for ax in 'xyzw']
    r_imu = R.from_quat(q_ros)
    
    r_basis_change = R.from_euler('xyz', [-90, 0, -90], degrees=True)
    r_colmap = r_imu * r_basis_change

    q_final = r_colmap.as_quat()
    
    return {
        'header': {'stamp': {'sec': int(msg.header.stamp.sec), 'nanosec': int(msg.header.stamp.nanosec)}, 'frame_id': msg.header.frame_id},
        'orientation': {ax: float(getattr(msg.orientation, ax)) for ax in 'xyzw'},
        'colmap_orientation': {
            'w': float(q_final[3]),
            'x': float(q_final[0]),
            'y': float(q_final[1]),
            'z': float(q_final[2])
        },
        'angular_velocity': {ax: float(getattr(msg.angular_velocity, ax)) for ax in 'xyz'},
        'linear_acceleration': {ax: float(getattr(msg.linear_acceleration, ax)) for ax in 'xyz'}
    }

# --- Main Sync Loop ---

def synchronize(bag_path, target_fps, cam_topics, cam_info_topics, imu_topic, gps_topic):
    all_topics = cam_topics + [gps_topic] + [imu_topic]
    bag_name = os.path.basename(bag_path.rstrip('/'))
    output_dir = os.path.join("exported_data", bag_name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    extract_camera_configs(bag_path, cam_info_topics, os.path.join(output_dir, "robotics_cam_config.yaml"))
    extract_rig_config(cam_topics, os.path.join(output_dir, "robotics_rig_config.json"))

    print(f"Indexing Bag: {bag_path}")
    reader = get_reader(bag_path, topics=all_topics)
    topic_indices = defaultdict(list)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, _, timestamp = reader.read_next()
        topic_indices[topic].append(timestamp)
    
    active_topics = {t: indices for t, indices in topic_indices.items() if indices}
    if not active_topics: return

    start_time, end_time = max(ts[0] for ts in active_topics.values()), min(ts[-1] for ts in active_topics.values())
    interval_ns, current_target, schedule = int(1e9 / target_fps), start_time, []

    print("Building Schedule...")
    while current_target <= end_time:
        slot = {}
        for topic, ts_list in active_topics.items():
            idx = bisect.bisect_left(ts_list, current_target)
            if topic == imu_topic or topic == gps_topic:
                if 0 < idx < len(ts_list):
                    t0, t1 = ts_list[idx-1], ts_list[idx]
                    slot[topic] = {'needed': [t0, t1], 'alpha': (current_target - t0) / (t1 - t0), 'mode': 'lerp', 'diag_ts': (t0 + t1) / 2}
                else:
                    ts = ts_list[0] if idx == 0 else ts_list[-1]
                    slot[topic] = {'needed': [ts], 'mode': 'closest', 'diag_ts': ts}
            else:
                if idx == 0: best_ts = ts_list[0]
                elif idx >= len(ts_list): best_ts = ts_list[-1]
                else:
                    t0, t1 = ts_list[idx-1], ts_list[idx]
                    best_ts = t1 if (t1 - current_target) < (current_target - t0) else t0
                slot[topic] = {'needed': [best_ts], 'mode': 'closest', 'diag_ts': best_ts}
        schedule.append((current_target, slot))
        current_target += interval_ns

    offset_feedback(schedule, cam_topics, imu_topic, gps_topic, output_dir)

    lookup = defaultdict(list)
    for slot_idx, (_, needed_dict) in enumerate(schedule):
        for topic, info in needed_dict.items():
            for ts in info['needed']: lookup[ts].append((slot_idx, topic, info))

    for t in all_topics: os.makedirs(get_data_path(t, output_dir, cam_topics, imu_topic, gps_topic), exist_ok=True)

    lerp_buffer, processed_slots = {}, set()
    data_reader = get_reader(bag_path, topics=all_topics)
    pbar = tqdm(total=len(schedule), desc="Exporting Bundles")

    while data_reader.has_next():
        topic, data, timestamp = data_reader.read_next()
        if topic not in all_topics or timestamp not in lookup: continue
        for slot_idx, target_topic, info in lookup[timestamp]:
            if target_topic != topic: continue
            msg = deserialize_message(data, get_message(topic_types[topic]))
            save_path = os.path.join(get_data_path(topic, output_dir, cam_topics, imu_topic, gps_topic), f"{slot_idx+1:04d}")
            if topic in cam_topics:
                cv2.imwrite(f"{save_path}.jpg", cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR))
            elif topic == imu_topic or topic == gps_topic:
                if info['mode'] == 'lerp':
                    key = (slot_idx, topic)
                    if key not in lerp_buffer: lerp_buffer[key] = msg
                    else:
                        m_stored = lerp_buffer.pop(key)
                        t0, m1 = (m_stored, msg) if (m_stored.header.stamp.sec * 1e9 + m_stored.header.stamp.nanosec) < (msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec) else (msg, m_stored)
                        interp = interpolate_gps(t0, m1, info['alpha']) if topic == gps_topic else interpolate_imu(t0, m1, info['alpha'])
                        with open(f"{save_path}.yaml", 'w') as f: yaml.dump(gps_to_dict(interp) if topic == gps_topic else imu_to_dict(interp), f, sort_keys=False)
                else:
                    with open(f"{save_path}.yaml", 'w') as f: yaml.dump(gps_to_dict(msg) if topic == gps_topic else imu_to_dict(msg), f, sort_keys=False)
            if slot_idx not in processed_slots:
                processed_slots.add(slot_idx)
                pbar.update(1)
    pbar.close()

    print(f"Export complete: {output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rosbag', type=str, required=True)
    parser.add_argument('--hz', type=float, default=3.0)
    args = parser.parse_args()

    CAM_TOPICS = [
        #"/zed_multi/zed_front/right/image_rect_color/compressed", 
        "/zed_multi/zed_front/left/image_rect_color/compressed",
        #"/zed_multi/zed_rear/right/image_rect_color/compressed", 
        "/zed_multi/zed_rear/left/image_rect_color/compressed",
        #"/zed_multi/zed_right/right/image_rect_color/compressed", 
        "/zed_multi/zed_right/left/image_rect_color/compressed",
        #"/zed_multi/zed_left/right/image_rect_color/compressed", 
        "/zed_multi/zed_left/left/image_rect_color/compressed",
    ]
    
    IMU_TOPIC = "/zed_multi/zed_front/imu/data"
    # IMU_TOPICS = ["/zed_multi/zed_front/imu/data", 
    #               "/zed_multi/zed_rear/imu/data", 
    #               "/zed_multi/zed_right/imu/data", 
    #               "/zed_multi/zed_left/imu/data"]
    
    GPS_TOPIC = "/fix"
    
    INFO_TOPICS = [t.replace("image_rect_color/compressed", "camera_info") for t in CAM_TOPICS]

    synchronize(args.rosbag, args.hz, CAM_TOPICS, INFO_TOPICS, IMU_TOPIC, GPS_TOPIC)