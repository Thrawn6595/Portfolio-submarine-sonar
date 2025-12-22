from setuptools import setup, find_packages

setup(
    name="ml_toolkit",
    version="1.0.0",
    description="Reusable ML utilities with Economist styling",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0"
    ],
    python_requires=">=3.9"
)
