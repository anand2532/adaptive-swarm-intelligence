from setuptools import setup, find_packages

setup(
    name="drone-swarm-simulation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "torch>=1.9.0",
        "gymnasium>=0.26.0",
        "pyyaml>=5.4.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "pylint>=2.8.0",
            "mypy>=0.900"
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A PSO-RL hybrid simulation framework for autonomous drone swarms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="drone, swarm, reinforcement-learning, particle-swarm-optimization",
    url="https://github.com/yourusername/drone-swarm-simulation",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "drone-sim=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config.yaml"],
    },
)