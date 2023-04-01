from setuptools import setup, find_packages

setup(
    name='house_price_prediction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        ...
    ],
    entry_points={
        'console_scripts': [
            'house_price_prediction=src.main:main',
        ],
    },
)
