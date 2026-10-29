import os
from setuptools import find_packages, setup
from glob import glob
package_name = 'ekf_2602_foxtrot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'config'), glob('config/*.*')),
        (os.path.join('share', package_name,'launch'), glob('launch/*.*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='juan-londono',
    maintainer_email='juanjoselondonodavid@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
entry_points={
        'console_scripts': [
            'ekf_node_2602_foxtrot = ekf_2602_foxtrot.ekf_node_2602_foxtrot:main',
            'covariance_relay = ekf_2602_foxtrot.covariance_relay:main',
        ],
    },
)
