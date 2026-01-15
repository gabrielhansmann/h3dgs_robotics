import bisect
from scipy.spatial.transform import Slerp
import numpy as np
import yaml
import os
import cv2
from collections import defaultdict
from tqdm import tqdm
import argparse

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Imu, NavSatFix
from scipy.spatial.transform import Rotation as R

# Configuration
BAG_PATH = "rosbag2_2026_01_13-14_52_54"
OUTPUT_PATH = "exported_data"

TARGET_FPS = 3.0
GPS_TOPIC = "/fix"
IMU_TOPICS = [
    "/zed_multi/zed_front/imu/data",
    "/zed_multi/zed_rear/imu/data",
    "/zed_multi/zed_right/imu/data",
    "/zed_multi/zed_left/imu/data",
]
CAM_TOPICS = [
    "/zed_multi/zed_front/right/image_rect_color/compressed",
    "/zed_multi/zed_front/left/image_rect_color/compressed",
    "/zed_multi/zed_rear/right/image_rect_color/compressed",
    "/zed_multi/zed_rear/left/image_rect_color/compressed",
    "/zed_multi/zed_right/right/image_rect_color/compressed",
    "/zed_multi/zed_right/left/image_rect_color/compressed",
    "/zed_multi/zed_left/right/image_rect_color/compressed",
    "/zed_multi/zed_left/left/image_rect_color/compressed",
]
ALL_TOPICS = CAM_TOPICS + [GPS_TOPIC] + IMU_TOPICS

def lerp(v0, v1, alpha):
    return v0 + alpha * (v1 - v0)

def interpolate_imu(m0, m1, alpha):
    out = Imu()
    # Interpolate the timestamp in the header as well
    t0_ns = m0.header.stamp.sec * 1e9 + m0.header.stamp.nanosec
    t1_ns = m1.header.stamp.sec * 1e9 + m1.header.stamp.nanosec
    target_ns = int(lerp(t0_ns, t1_ns, alpha))
    
    out.header.frame_id = m0.header.frame_id
    out.header.stamp.sec = int(target_ns // 1e9)
    out.header.stamp.nanosec = int(target_ns % 1e9)

    # Correct way to set nested attributes in ROS2 messages
    for field_name in ['linear_acceleration', 'angular_velocity']:
        target_field = getattr(out, field_name)
        m0_field = getattr(m0, field_name)
        m1_field = getattr(m1, field_name)
        for axis in ['x', 'y', 'z']:
            val = lerp(getattr(m0_field, axis), getattr(m1_field, axis), alpha)
            setattr(target_field, axis, val)
    
    # Orientation Slerp
    q0 = [m0.orientation.x, m0.orientation.y, m0.orientation.z, m0.orientation.w]
    q1 = [m1.orientation.x, m1.orientation.y, m1.orientation.z, m1.orientation.w]
    
    # Using scipy for robust Slerp
    rot = R.from_quat([q0, q1])
    slerp_obj = Slerp([0, 1], rot)
    interp_rot = slerp_obj([alpha])[0]
    
    q_out = interp_rot.as_quat()
    out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w = q_out
    return out

def interpolate_gps(m0, m1, alpha):
    out = NavSatFix()
    # Interpolate timestamp
    t0_ns = m0.header.stamp.sec * 1e9 + m0.header.stamp.nanosec
    t1_ns = m1.header.stamp.sec * 1e9 + m1.header.stamp.nanosec
    target_ns = int(lerp(t0_ns, t1_ns, alpha))
    
    out.header.frame_id = m0.header.frame_id
    out.header.stamp.sec = int(target_ns // 1e9)
    out.header.stamp.nanosec = int(target_ns % 1e9)
    
    out.status = m0.status
    out.latitude = lerp(m0.latitude, m1.latitude, alpha)
    out.longitude = lerp(m0.longitude, m1.longitude, alpha)
    out.altitude = lerp(m0.altitude, m1.altitude, alpha)
    # Optional: Interpolate covariance if needed
    out.position_covariance = m0.position_covariance 
    return out

def offset_feedback(schedule):
    topic_jitter_ms = defaultdict(list)

    for target_time, needed_timestamps in schedule:
        for top, ts_info in needed_timestamps.items():
            actual_ts = ts_info['needed'][0]
            offset_ms = abs(actual_ts - target_time) / 1e6
            topic_jitter_ms[top].append(offset_ms)

    print("\n" + "="*50)
    print(f"{'TOPIC TYPE':<15} | {'AVG OFFSET (ms)':<17} | {'MAX OFFSET (ms)':<15}")
    print("-" * 50)

    def print_group_stats(name, topics):
        group_offsets = []
        for t in topics:
            if t in topic_jitter_ms:
                group_offsets.extend(topic_jitter_ms[t])
        
        if group_offsets:
            avg_val = sum(group_offsets) / len(group_offsets)
            max_val = max(group_offsets)
            print(f"{name:<15} | {avg_val:>14.3f} ms | {max_val:>11.3f} ms")
        else:
            print(f"{name:<15} | {'No Data':>18} | {'N/A':>15}")

    # Print Categorized Results
    print_group_stats("CAMERAS", CAM_TOPICS)
    print_group_stats("IMUs", IMU_TOPICS)
    print_group_stats("GPS", [GPS_TOPIC])
    print("="*50)

def synchronize():
    print("Indexing timestamps...")
    reader = get_reader(BAG_PATH)
    topic_indices = {topic: [] for topic in ALL_TOPICS}
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    
    while reader.has_next():
        topic, _, timestamp = reader.read_next()
        if topic in topic_indices: topic_indices[topic].append(timestamp)
    del reader

    active_topics = {t: indices for t, indices in topic_indices.items() if indices}
    if len(active_topics) < len(ALL_TOPICS):
        missing = set(ALL_TOPICS) - set(active_topics.keys())
        print(f"Warning: Missing data for topics: {missing}")
    start_time = max(ts_list[0] for ts_list in active_topics.values())
    end_time = min(ts_list[-1] for ts_list in active_topics.values())
    
    interval_ns = int(1e9 / TARGET_FPS)
    current_target = start_time
    schedule = [] 

    while current_target <= end_time:
        slot = {}
        for topic, ts_list in active_topics.items():
            idx = bisect.bisect_left(ts_list, current_target)
            
            if topic in IMU_TOPICS or topic == GPS_TOPIC:
                # interpolation of the closest two points for imu and gps
                if 0 < idx < len(ts_list):
                    t0, t1 = ts_list[idx-1], ts_list[idx]
                    alpha = (current_target - t0) / (t1 - t0)
                    slot[topic] = {'needed': [t0, t1], 'alpha': alpha, 'mode': 'lerp'}
                else:
                    slot[topic] = {'needed': [ts_list[idx if idx < len(ts_list) else -1]], 'mode': 'closest'}
            else:
                # find the closest for camera
                best_ts = ts_list[idx] if (idx < len(ts_list) and (idx == 0 or abs(ts_list[idx]-current_target) < abs(ts_list[idx-1]-current_target))) else ts_list[idx-1]
                slot[topic] = {'needed': [best_ts], 'mode': 'closest'}
        
        schedule.append((current_target, slot))
        current_target += interval_ns

    print(f"{len(schedule)} bundles...")

    offset_feedback(schedule)

    lookup = {}
    for slot_idx, (_, needed_dict) in enumerate(schedule):
        for topic, info in needed_dict.items():
            for ts in info['needed']:
                if ts not in lookup:
                    lookup[ts] = []
                lookup[ts].append((slot_idx, topic, info.get('mode'), info.get('alpha')))

    # dir structure
    main_dir = OUTPUT_PATH
    os.makedirs(main_dir, exist_ok=True)
    for topic in ALL_TOPICS:
        folder_name = topic.strip('/').replace('/', '_')
        os.makedirs(os.path.join(main_dir, folder_name), exist_ok=True)

    # buffer for interpolation
    lerp_buffer = {} # { (slot_idx, topic): first_msg }

    print(f"Exporting to {main_dir}...")
    data_reader = get_reader(BAG_PATH)
    pbar = tqdm(total=len(schedule), desc="Exporting Bundles", unit="bundle")
    current_slot_idx = 0

    while data_reader.has_next():
        topic, data, timestamp = data_reader.read_next()

        if timestamp in lookup:
            for slot_idx, target_topic, mode, alpha in lookup[timestamp]:
                current_slot_idx = slot_idx+1
                if target_topic != topic:
                    continue
                
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                
                # format: 0001, 0002,...
                file_base = f"{slot_idx+1:04d}"
                folder_name = topic.strip('/').replace('/', '_')
                save_path = os.path.join(main_dir, folder_name, file_base)

                # imgs
                if topic in CAM_TOPICS:
                    # CompressedImage to OpenCV
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    cv2.imwrite(f"{save_path}.png", cv_img)

                # imu/gps
                elif topic in IMU_TOPICS or topic == GPS_TOPIC:
                    if mode == 'lerp':
                        # interpolation
                        key = (slot_idx, topic)
                        if key not in lerp_buffer:
                            # save the first message (t0)
                            lerp_buffer[key] = msg
                        else:
                            # We have the second message (t1). Interpolate and save.
                            m0 = lerp_buffer.pop(key)
                            m1 = msg
                            if topic == GPS_TOPIC:
                                interp_msg = interpolate_gps(m0, m1, alpha)
                                data_to_save = gps_to_filtered_dict(interp_msg)
                            else:
                                interp_msg = interpolate_imu(m0, m1, alpha)
                                data_to_save = imu_to_filtered_dict(interp_msg)
                            with open(f"{save_path}.yaml", 'w') as f:
                                yaml.dump(data_to_save, f, default_flow_style=False, sort_keys=False)
                    else:
                        # closest match
                        if topic == GPS_TOPIC:
                            data_to_save = gps_to_filtered_dict(msg)
                        else:
                            data_to_save = imu_to_filtered_dict(msg)
                        with open(f"{save_path}.yaml", 'w') as f:
                            yaml.dump(gps_to_filtered_dict(msg), f, default_flow_style=False, sort_keys=False)
        if current_slot_idx > pbar.n:
            pbar.update(current_slot_idx - pbar.n)
    pbar.close()
    print(f"\nSuccessfully exported to {main_dir}")

def imu_to_filtered_dict(msg):
    """Cast NumPy scalars to python floats to prevent YAML 'multiarray' tags"""
    return {
        'header': {
            'stamp': {
                'sec': int(msg.header.stamp.sec),
                'nanosec': int(msg.header.stamp.nanosec)
            },
            'frame_id': msg.header.frame_id
        },
        'orientation': {
            'x': float(msg.orientation.x),
            'y': float(msg.orientation.y),
            'z': float(msg.orientation.z),
            'w': float(msg.orientation.w)
        },
        'angular_velocity': {
            'x': float(msg.angular_velocity.x),
            'y': float(msg.angular_velocity.y),
            'z': float(msg.angular_velocity.z)
        },
        'linear_acceleration': {
            'x': float(msg.linear_acceleration.x),
            'y': float(msg.linear_acceleration.y),
            'z': float(msg.linear_acceleration.z)
        }
    }

def gps_to_filtered_dict(msg):
    return {
        'header': {
            'stamp': {
                'sec': int(msg.header.stamp.sec),
                'nanosec': int(msg.header.stamp.nanosec)
            },
            'frame_id': msg.header.frame_id
        },
        'latitude': float(msg.latitude),
        'longitude': float(msg.longitude),
        'altitude': float(msg.altitude)
    }

def get_reader(path):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    return reader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rosbag', default=BAG_PATH, help='rosbag path')
    parser.add_argument('--output', default=OUTPUT_PATH, help='output path')
    args = parser.parse_args()
    
    BAG_PATH = args.rosbag
    OUTPUT_PATH = args.output

    synchronize()