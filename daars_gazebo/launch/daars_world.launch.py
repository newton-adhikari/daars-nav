"""Launch file for Gazebo with DAARS arena and TurtleBot3 Burger."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Paths
    daars_pkg = get_package_share_directory('daars_gazebo')
    tb3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    tb3_desc_pkg = get_package_share_directory('turtlebot3_description')

    world_file = os.path.join(daars_pkg, 'worlds', 'daars_arena.world')
    urdf_file = os.path.join(tb3_desc_pkg, 'urdf', 'turtlebot3_burger.urdf')

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    # Launch arguments
    x_pose = LaunchConfiguration('x_pose', default='1.0')
    y_pose = LaunchConfiguration('y_pose', default='1.0')

    # Gazebo server + client
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items(),
    )

    # Robot state publisher
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}],
    )

    # Spawn robot inside the arena at (1, 1)
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'burger',
            '-file', os.path.join(tb3_gazebo_pkg, 'models',
                                   'turtlebot3_burger', 'model.sdf'),
            '-x', x_pose,
            '-y', y_pose,
            '-z', '0.01',
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('x_pose', default_value='1.0'),
        DeclareLaunchArgument('y_pose', default_value='1.0'),
        gazebo,
        robot_state_pub,
        spawn_robot,
    ])

 