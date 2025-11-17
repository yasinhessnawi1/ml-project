"""
Setup script for ML Project installation.

This allows the project to be installed as a package, making imports cleaner.

Installation:
    pip install -e .           # Development/editable mode
    pip install .              # Regular installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
# Filter out comments and empty lines
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="ml-project",
    version="1.0.0",
    author="Yasin Hessnawi",
    author_email="yasin.hessnawi@example.com",
    description="Multimodal Genre Classification with Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yasinhessnawi/ml-project",
    project_urls={
        "Bug Tracker": "https://github.com/yasinhessnawi/ml-project/issues",
        "Documentation": "https://github.com/yasinhessnawi/ml-project/tree/main/.docs",
    },
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-train=scripts.train:main",
            "ml-evaluate=scripts.evaluate:main",
            "ml-test=test_implementation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
