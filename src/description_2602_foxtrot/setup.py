from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'description_2602_foxtrot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'diffdrive_urdf'), glob('diffdrive_urdf/*.*')),
        (os.path.join('share', package_name,'ackermann_urdf'), glob('ackermann_urdf/*.*')),
    ],
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
        ],
    },
)
