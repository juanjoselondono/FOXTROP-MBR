from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sysid_2602_foxtrot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ddrozoa8',
    maintainer_email='david.rozo31@eia.edu.co',
    description='System identification pipeline for RC car speed estimation using VICON ground truth',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'speed_estimator_node = sysid_2602_foxtrot.speed_estimator_node:main',
            'throttle_profile_node = sysid_2602_foxtrot.throttle_profile_node:main',
            'manual_cruise_node = sysid_2602_foxtrot.manual_cruise_node:main',
        ],
    },
)
