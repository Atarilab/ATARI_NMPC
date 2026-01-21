from setuptools import setup, find_packages

setup(
    name="traj_opt_acados",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
        # For example:
        # 'numpy',
        # 'scipy',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of your package",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)