"""Distribution setup."""

import versioneer
from setuptools import setup, find_packages

with open("README.rst", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="CORAL",
    author="Jake Nunemaker",
    description=long_description,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    install_requires=[
        "numpy",
        "pandas",
        "marmot-agents>=0.2.5",
        "orbit-nrel>=1.0.6",
        "matplotlib",
        "jupyterlab",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pylint",
            "flake8",
            "black",
            "isort",
            "pytest",
            "sphinx",
            "sphinx-rtd-theme",
        ]
    },
)
