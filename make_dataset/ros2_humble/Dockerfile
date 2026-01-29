FROM osrf/ros:humble-desktop

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies + ROS packages
RUN apt update && apt upgrade -y && apt install -y \
    nano \
    tmux \
    python3-pip \
    python3-opencv \
    python3-yaml \
    python3-numpy \
    ros-humble-slam-toolbox \
    ros-humble-pointcloud-to-laserscan \
    ros-humble-rosbag2 \
    ros-humble-gps-msgs \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-foxglove-msgs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Source ROS automatically
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

CMD ["bash"]
