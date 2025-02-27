from setuptools import setup, find_packages

setup(
    name='pyrf',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',],
    author='Mark Hobbs',
    author_email='markhobbs91@gmail.com',
    description='A lightweight Python package with minimal dependencies for generating spatially correlated random fields',
    url='https://github.com/mark-hobbs/pyrf',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)