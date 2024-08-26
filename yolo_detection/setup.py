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
                'debug_node = yolo_detection.debug_node:main',
        ],
    },
)
