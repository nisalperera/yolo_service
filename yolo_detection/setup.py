import os
import sys
import pathlib
import subprocess

from glob import glob
from setuptools import setup

package_name = 'yolo_detection'

__base__ = pathlib.Path(__file__).parent.resolve()

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', os.path.join(__base__, "requirements.txt")])

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', "opencv-python", "typing-extensions", "ultralytics", "lap"],
    zip_safe=True,
    maintainer='Nisal Chinthana Perera',
    maintainer_email='chinthanapereranisal@gmail.com',
    description='YOLO package for ROS2',
    license='Apache 2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
                'yolo_node = yolo_detection.yolo_node:main',
                # 'debug_node = yolo_detection.debug_node:main',
        ],
    },
)
