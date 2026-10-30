from setuptools import find_packages, setup

package_name = 'perception_2602_foxtrot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='juan-londono',
    maintainer_email='juanjoselondonodavid@gmail.com',
    description='2D LiDAR perception: line segment extraction (split-and-merge)',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'line_extraction_node = perception_2602_foxtrot.line_extraction_node:main',
        ],
    },
)
