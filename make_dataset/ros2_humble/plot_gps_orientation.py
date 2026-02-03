import argparse
import os
import yaml
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * (limits[:, 1] - limits[:, 0]).max()
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

def plot_trajectory(root, step=20, use_colmap=False):
    if not os.path.exists(root): 
        raise NotADirectoryError(f'Directory {root} does not exist')

    gps_dir = os.path.join(root, 'inputs', 'robotics_gps')
    imu_dir = os.path.join(root, 'inputs', 'robotics_imu')

    yaml_files = sorted(glob.glob(os.path.join(gps_dir, "*.yaml")))
    
    gps_coords = {'x': [], 'y': [], 'z': []}
    vec_z = {'u': [], 'v': [], 'w': []} 
    vec_y = {'u': [], 'v': [], 'w': []} 
    vec_x = {'u': [], 'v': [], 'w': []} 

    print(f"Plotting 3D Trajectory...")
    print(f"Frame Convention: {'COLMAP (RDF: Z-fwd, Y-down)' if use_colmap else 'Normal (FLU: X-fwd, Z-up)'}")

    for file_path in yaml_files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r') as f:
            gps_val = yaml.safe_load(f)
            gps_coords['x'].append(gps_val['x'])
            gps_coords['y'].append(gps_val['y'])
            gps_coords['z'].append(gps_val['z'])

        imu_file = os.path.join(imu_dir, filename)
        if os.path.exists(imu_file):
            with open(imu_file, 'r') as f:
                imu_val = yaml.safe_load(f)
            
            key = 'colmap_orientation' if use_colmap else 'orientation'
            if key not in imu_val:
                print(f"Warning: {key} not found in {filename}. Skipping orientation for this frame.")
                continue
                
            q = imu_val[key]
            rotation = R.from_quat([q['x'], q['y'], q['z'], q['w']])
            
            # Local Z (Blue)
            z_world = rotation.apply([0, 0, 1])
            vec_z['u'].append(z_world[0]); vec_z['v'].append(z_world[1]); vec_z['w'].append(z_world[2])

            # Local Y (Green)
            y_world = rotation.apply([0, 1, 0])
            vec_y['u'].append(y_world[0]); vec_y['v'].append(y_world[1]); vec_y['w'].append(y_world[2])

            # Local X (Red)
            x_world = rotation.apply([1, 0, 0])
            vec_x['u'].append(x_world[0]); vec_x['v'].append(x_world[1]); vec_x['w'].append(x_world[2])

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gps_coords['x'], gps_coords['y'], gps_coords['z'], color='black', alpha=0.3, label='GPS Path')
    
    # 1. Plot Z-axis (Blue)
    # COLMAP: Forward (Front) | Normal: Up (Sky)
    ax.quiver(gps_coords['x'][::step], gps_coords['y'][::step], gps_coords['z'][::step],
              vec_z['u'][::step], vec_z['v'][::step], vec_z['w'][::step],
              length=1.5, color='blue', label='Z-axis (Blue)')
    
    # 2. Plot Y-axis (Green)
    # COLMAP: Down (Ground) | Normal: Left
    ax.quiver(gps_coords['x'][::step], gps_coords['y'][::step], gps_coords['z'][::step],
              vec_y['u'][::step], vec_y['v'][::step], vec_y['w'][::step],
              length=1.0, color='green', label='Y-axis (Green)')

    # 3. Plot X-axis (Red)
    # COLMAP: Right | Normal: Forward (Front)
    ax.quiver(gps_coords['x'][::step], gps_coords['y'][::step], gps_coords['z'][::step],
              vec_x['u'][::step], vec_x['v'][::step], vec_x['w'][::step],
              length=1.0, color='red', label='X-axis (Red)')
    
    set_axes_equal(ax)
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_zlabel('World Z')
    ax.set_title(f'Coordinate System Check\nCOLMAP Mode: {use_colmap}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', required=True, help='Path to your data folder')
    parser.add_argument('--step', type=int, default=10, help='Density of arrows')
    parser.add_argument('--colmap', action='store_true', help='Use colmap_orientation keys')
    
    args = parser.parse_args()
    plot_trajectory(args.project_dir, step=args.step, use_colmap=args.colmap)