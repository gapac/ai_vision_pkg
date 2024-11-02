from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ai_vision_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gapac',
    maintainer_email='gasper.jezernik001@gmail.com',
    description='Package that implements grounded AI vision tasks using SAM, Florence, GroudingDino...',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = ai_vision_pkg.publisher_member_function:main',
            'listener = ai_vision_pkg.subscriber_member_function:main',
            'florence = ai_vision_pkg.florence_inference_member_function:main',
            'image_publisher = ai_vision_pkg.image_publisher:main',
            'sam2 = ai_vision_pkg.sam2_inference:main',
        ],
    },
)
