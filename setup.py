from setuptools import setup, find_packages

setup(
    name="try_on_dataset",  # Replace with your desired package name
    version="0.1.0",
    description="A package for processing single image try-on datasets",
    author="Ali Hassan",  # Replace with your name
    author_email="alyhassan862@gmail.com",  # Replace with your email
    url="https://github.com/alihssan/TPD-packaged.git",  # Replace with your repository URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires='==3.9',  # Ensure compatibility with the Python version in environment.yml
)
