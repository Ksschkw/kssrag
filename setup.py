from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="kssrag",
    version="0.4.0",
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords="rag, retrieval, generation, ai, nlp, faiss, bm25",
    package_dir={"": "."},
    packages=find_packages(where="."),
    # NOTE: code uses PEP 604 unions (`str | List[str]`) and builtin generics,
    # which require Python 3.10+.
    python_requires=">=3.10, <4",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "anyio>=3.7.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "numpy>=1.26.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.2.0",  # enable_decoding in SettingsConfigDict
        "rapidfuzz>=3.0.0",
        "python-multipart>=0.0.6",
        "pypdf>=3.0.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "bm25s>=0.2.14",  # Add bm25s
        "stemmer>=0.0.4",
        "pystemmer>=3.0.0",
        "python-docx>=1.1.0",
        "openpyxl>=3.1.5", 
        "python-pptx>=1.0.2",
    ],

    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy", "openai>=1.0.0"],
        "gpu": ["faiss-gpu>=1.7.0"],
        # FAISS / semantic (hybrid_online, faiss) support: heavy, optional.
        "faiss": [
            "faiss-cpu>=1.7.0; sys_platform != 'win32'",
            "faiss-cpu-win; sys_platform == 'win32'",
            "sentence-transformers>=3.0.0",
        ],
        "all": ["faiss-gpu>=1.7.0", "sentence-transformers[gpu]"],

        'ocr': [
        'paddleocr>=2.7.0',
        'paddlepaddle>=2.5.0', 
        'pytesseract>=0.3.10',
        'Pillow>=10.0.0',
        'opencv-python>=4.8.0'
        ],
        'office': [
        'python-docx>=1.1.0',
        'openpyxl>=3.1.0',
        'python-pptx>=0.6.23',
        ],
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