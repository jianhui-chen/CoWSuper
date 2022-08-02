from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()
    
setup(
    name='cowsupervision',
    version='0.7.1.2',
    description='A friendly watermarking tool with optional GUI component.',
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='holypython.com',
    author_email='watermarkd@holypython.com',
    url="https://holypython.com/",
    download_url = 'https://github.com/holypython/Watermarkd/archive/0.7.1.2.tar.gz',
    packages=['Watermarkd'],

    install_requires=[
       'pillow',
       'pysimplegui',
    ],

    python_requires='>=3.6'
)