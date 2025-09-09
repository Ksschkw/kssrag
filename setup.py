from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="kssrag",
    version="0.1.0",
    description="A flexible Retrieval-Augmented Generation framework by Ksschkw",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ksschkw/kssrag",
    author="Ksschkw",
    author_email="kookafor893@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords="rag, retrieval, generation, ai, nlp, faiss, bm25",
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8, <4",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "rank-bm25>=0.2.2",
        "numpy>=1.26.0",
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=3.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "rapidfuzz>=3.0.0",
        "python-multipart>=0.0.6",
        "pypdf>=3.0.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],

    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "gpu": ["faiss-gpu>=1.7.0"],
        "all": ["faiss-gpu>=1.7.0", "sentence-transformers[gpu]"]
    },
    entry_points={
        "console_scripts": [
            "kssrag=kssrag.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/Ksschkw/kssrag/issues",
        "Source": "https://github.com/Ksschkw/kssrag",
        "Documentation": "https://github.com/Ksschkw/kssrag/docs",
    },
)