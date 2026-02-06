from setuptools import setup, find_packages

setup(
    name='gym-pps',
    version='0.1.0',
    description='Predator-Prey Swarm environment compatible with Gymnasium',
    author='',
    packages=find_packages(),
    package_dir={'gym_pps': 'gym_pps'},
    install_requires=[
        'gymnasium>=0.29.0',
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'scipy>=1.4.1',
        'pygame>=2.1.0',
    ],
    extras_require={
        'test': ['pytest'],
    },
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
