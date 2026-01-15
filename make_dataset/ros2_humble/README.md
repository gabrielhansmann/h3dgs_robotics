### Install Docker environment for ros2 humble
1. pull docker image from dockerhub
```bash
docker build -t ros2humble_h3dgs_robotics .
```
2. run the docker image
```bash
xhost +local:docker

docker run -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  ros2humble_h3dgs_robotics:latest \
  bash
```
3. source ros inside the container
```bash
source /opt/ros/humble/setup.bash
```

if there is a problem with ownership:
```bash
sudo chown -R $USER:$USER <DIR>
```
