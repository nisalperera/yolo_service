import os

from glob import glob
from setuptools import setup

package_name = 'yolo_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py'))
    ],
    install_requires=["opencv-python>=4.5.5.64", "typing-extensions>=4.8.0", "ultralytics>=8.2.103", "lap>=0.3.0"],
    setup_requires=["opencv-python>=4.5.5.64", "typing-extensions>=4.8.0", "ultralytics>=8.2.103", "lap>=0.3.0"],
    zip_safe=True,
    maintainer='Nisal Chinthana Perera',
    maintainer_email='chinthanapereranisal@gmail.com',
    description='YOLO package for ROS2',
    license='Apache 2.0',
    # extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
                'detector = yolo_detection.yolo_node:main',
                'visualizer = yolo_detection.viz_node:main',
        ],
    },
)
