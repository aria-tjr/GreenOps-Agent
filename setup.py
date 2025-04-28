from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="greenops-agent",
    version="0.1.0",
    description="Optimize Kubernetes workloads for energy efficiency and carbon footprint",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/greenops-agent",
    license="MIT",
    author="GreenOps Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "numpy>=1.20.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "prometheus-client>=0.17.0",
    ],
    extras_require={
        "ml": [
            "tensorflow>=2.12.0; platform_machine != 'arm64' and platform_machine != 'aarch64'",
            "tensorflow-macos>=2.12.0; platform_machine == 'arm64' or platform_machine == 'aarch64'",
            "scikit-learn>=1.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "pylint>=2.17.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "greenops-agent=greenops_agent.main:run_cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Build Tools",
    ],
)