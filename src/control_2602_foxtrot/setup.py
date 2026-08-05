from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'control_2602_foxtrot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.*')),
        # Route the script directly to the lib directory where ROS 2 expects executables
        ('lib/' + package_name, [
            'AEB/aeb_ttc.py',
            'AEB/wall_follower.py',
            'AEB/supervisor.py', 
            'AEB/lane_keeper.py'
        ]),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ddrozoa8',
    maintainer_email='david.rozo31@eia.edu.co',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # Left blank because we are copying the script directly via data_files
        ],
    },
)