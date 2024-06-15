from setuptools import setup, find_packages

setup(
    name='QSOLens',
    version='0.2',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'keras'
    ],
    entry_points={
        'console_scripts': [
            'mycommand=QSOLens.NetworkArchitecture:predict',
        ],
    },
)