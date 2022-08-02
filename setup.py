from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()
    
setup(
    name='CoWSuper',
    version='0.0.1',
    description='The package produces labels using constrained weak supervision.',
    license="MIT License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Bert Huang',
    author_email='bert@cs.tufts.edu',
    packages=['CoWSuper'],

    install_requires=[
        'Tensorflow',
        'numpy',
        'Scikit-learn'
    ],

    python_requires='>=3.6'
)