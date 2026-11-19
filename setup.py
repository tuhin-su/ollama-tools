from setuptools import setup, find_packages

setup(
    name="ollama-tools",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "ollama-tools=ollama_tools.cli:main",
            "ollama-export=ollama_tools.cli:main_export",
            "ollama-import=ollama_tools.cli:main_import",
        ],
    },
)
