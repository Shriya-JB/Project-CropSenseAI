from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "CropSenseAI — AI System for Crop Classification, Stress Detection, and Irrigation Management"

setup(
    name="cropsenseai",
    version="1.0.0",
    author="CropSenseAI Team",
    author_email="team@cropsenseai.com",
    description="AI System for Crop Classification, Stress Detection, and Irrigation Management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cropsenseai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "opencv-python>=4.7.0",
        "tensorflow>=2.12.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "Pillow>=10.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pylint>=3.0.0",
            "jupyter>=1.0.0",
            "sphinx>=7.2.0",
        ],
        "api": [
            "fastapi>=0.103.0",
            "uvicorn>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cropsenseai=cropsenseai.dashboard.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
