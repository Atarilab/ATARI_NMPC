# Base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set noninteractive environment for apt
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG en_US.UTF-8

# Update and install system-related packages
RUN apt-get update && apt-get install -y \
    locales \
    cmake \
    make \
    nano \
    vim \
    curl \
    git \
    pip \
    software-properties-common \
    build-essential \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libzmq3-dev \
    '^libxcb.*-dev' \
    libx11-dev \
    libx11-xcb-dev \
    libxi-dev \
    libxrender-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libglu1-mesa-dev \
    python3-tk \
    libglfw3-dev \
    kbd \
    libeigen3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Configure locale
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

# Create a non-root user
ARG USERNAME=atari
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USERNAME \
    && mkdir -p /home/$USERNAME/.config \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

# Install sudo and grant sudo permissions
RUN apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER $USERNAME

# User permissions
RUN sudo chown -R $USERNAME:$USERNAME /home/$USERNAME/

# Create workspace directory
RUN mkdir -p /home/$USERNAME/workspace/
WORKDIR /home/$USERNAME/workspace/

# Install ROS 2 (Humble)
ENV ROS_DISTRO=humble

# Install ROS 2 dependencies
RUN sudo apt-get update && sudo apt-get upgrade -y && sudo apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    && sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && sudo apt-get update && sudo apt-get install -y \
    ros-$ROS_DISTRO-desktop \
    ros-$ROS_DISTRO-ament-cmake \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-colcon-mixin \
    python3-vcstool \
    ros-$ROS_DISTRO-gazebo-ros-pkgs \
    ros-$ROS_DISTRO-xacro \
    ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
    ros-$ROS_DISTRO-rosidl-generator-dds-idl \
    ros-$ROS_DISTRO-joint-state-publisher-gui \
    ros-$ROS_DISTRO-plotjuggler-ros \
    ros-dev-tools \
    && sudo rosdep init \
    && rosdep update

# Set the working directory
WORKDIR /home/$USERNAME/

# Install pinocchio from robotpkg repository
RUN echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list && \
    sudo curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -

# Update package list and install robotpkg-py310-pinocchio
RUN sudo apt-get update -y && \
    sudo apt-get install -qqy robotpkg-py310-pinocchio

# Set environment variables
ENV PATH=/opt/openrobots/bin:$PATH
ENV PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH
ENV CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH

# Create fake dist-info for pip to recognise the package
ENV PIN_VERSION=3.3.1
RUN sudo mkdir -p /opt/openrobots/lib/python3.10/site-packages/pin-${PIN_VERSION}.dist-info && \
    sudo chmod -R a+w /opt/openrobots/lib/python3.10/site-packages/pin-${PIN_VERSION}.dist-info && \
    sudo bash -c 'echo "Metadata-Version: 2.1\
    Name: pin\
    Version: '${PIN_VERSION}'\
    Summary: Installed via robotpkg\
    " > /opt/openrobots/lib/python3.10/site-packages/pin-'${PIN_VERSION}'.dist-info/METADATA'

# Same for casadi
ENV CASADI_VERSION=3.6.7
RUN sudo mkdir -p /opt/openrobots/lib/python3.10/site-packages/casadi-${CASADI_VERSION}.dist-info && \
    sudo chmod -R a+w /opt/openrobots/lib/python3.10/site-packages/casadi-${CASADI_VERSION}.dist-info && \
    sudo bash -c 'echo "Metadata-Version: 0.1\
    Name: casadi\
    Version: '${CASADI_VERSION}'\
    Summary: Installed via robotpkg\
    " > /opt/openrobots/lib/python3.10/site-packages/casadi-'${CASADI_VERSION}'.dist-info/METADATA'

# (Optional) Source .bashrc if needed for subsequent commandsp
RUN echo "source ~/.bashrc" >> ~/.bash_profile

# Create a virtual environment
RUN sudo apt install -y python3.10-venv
RUN python3 -m venv /home/$USERNAME/.venv
RUN /bin/bash -c "source /home/$USERNAME/.venv/bin/activate && \
     pip install --upgrade pip"

# Activate the virtual environment and install packages
RUN /bin/bash -c "source /home/$USERNAME/.venv/bin/activate && \
    pip install --upgrade \
    numpy==1.24 \
    typeguard==4.4.2 \
    scipy==1.8 \
    matplotlib==3.8 \
    colcon-core==0.18.4 \
    colcon-common-extensions==0.3.0 \
    lark==1.2.2 \
    catkin-pkg==1.0.0 \
    empy==3.3.4 \
    mujoco==3.2.7 \
    opencv-python==4.11.0.86 \
    pygame==2.6.1 \
    tqdm==4.67.1 \
    robot_descriptions==1.14 \
    psutil==7.0.0 \
    pyvicon-datastream \
    "

# Clone and install unitree_ros2 + CycloneDDS
RUN git clone https://github.com/unitreerobotics/unitree_ros2
WORKDIR /home/$USERNAME/unitree_ros2/cyclonedds_ws/src
RUN git clone https://github.com/ros2/rmw_cyclonedds -b $ROS_DISTRO && \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
WORKDIR /home/$USERNAME/unitree_ros2/cyclonedds_ws/
RUN bash -c "source /home/$USERNAME/.venv/bin/activate && \
             colcon build --packages-select cyclonedds"
RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
             source /home/$USERNAME/.venv/bin/activate && \
             colcon build"
RUN echo "source /home/$USERNAME/unitree_ros2/cyclonedds_ws/install/setup.sh" >> /home/$USERNAME/.bashrc

WORKDIR /home/$USERNAME

# Install unitree_sdk2_python
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && \
    cd unitree_sdk2_python && \
    /bin/bash -c "source /home/$USERNAME/.venv/bin/activate && \
                  pip install -e ."

# Install unitree_mujoco
RUN git clone https://github.com/unitreerobotics/unitree_mujoco.git

# Clone and install acados with Python interface
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --recursive --init && \
    mkdir -p build && cd build && \
    cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OPENMP=ON -DACADOS_NUM_THREADS=8 .. && \
    make install -j$(nproc) && \
    /bin/bash -c "source /home/$USERNAME/.venv/bin/activate && \
    pip install ../interfaces/acados_template"

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USERNAME/acados/lib"
ENV ACADOS_SOURCE_DIR="/home/$USERNAME/acados"

# Clone and install mj_pin_utils
RUN git clone https://github.com/Atarilab/mj_pin_utils.git && \
    /bin/bash -c "source /home/$USERNAME/.venv/bin/activate && \
    pip install -e ./mj_pin_utils"

# Set the final working directory
WORKDIR /home/$USERNAME/workspace/

# Source the virtual environment
RUN echo "source /home/$USERNAME/.venv/bin/activate" >> /home/$USERNAME/.bashrc
# Source ROS 2 Environments
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc 
# RUN echo "source /home/$USERNAME/ros2-controller/install/setup.bash" >> /home/$USERNAME/.bashrc 

CMD ["bash"]