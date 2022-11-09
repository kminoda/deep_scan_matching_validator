import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from rosbags.rosbag2 import Reader, Writer
from rosbags.serde import deserialize_cdr, serialize_cdr
from rosbags.typesys import get_types_from_idl, get_types_from_msg, register_types

from pathlib import Path
from tqdm import tqdm

from utils.read_pointcloud2 import read_points

NDT_PCD_TOPIC = '/localization/pose_estimator/points_aligned'
EKF_ODOM_TOPIC = '/localization/kinematic_state'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rosbag', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    odom_list = []
    odom_timestamp_list = []
    with Reader(args.rosbag) as reader:
        for connection, timestamp, raw_data in tqdm(reader.messages()):
            if connection.topic == EKF_ODOM_TOPIC:
                msg = deserialize_cdr(raw_data, connection.msgtype)
                odom_timestamp_list.append(
                    msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                )
                odom_list.append([
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ])
    odom_list = np.array(odom_list)
    odom_timestamp_list = np.array(odom_timestamp_list)

    scan_list = []
    count = 0
    with Reader(args.rosbag) as reader:
        for connection, timestamp, raw_data in tqdm(reader.messages()):
            if connection.topic == NDT_PCD_TOPIC:
                msg = deserialize_cdr(raw_data, connection.msgtype)
                points = np.array(list(read_points(msg)))
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                nearest_idx = (np.abs(odom_timestamp_list - timestamp)).argmin()
                nearest_pose = odom_list[nearest_idx]

                scan_data = {
                    'scan': points,
                    'pose': nearest_pose,
                }
                
                save_file = Path(args.output) / 'scan_data_{:07d}.pickle'.format(count)
                with open(save_file, mode="wb") as f:
                    pickle.dump(scan_data, f)

                count += 1
