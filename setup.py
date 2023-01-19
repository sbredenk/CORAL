"""Distribution setup."""

import versioneer
from setuptools import setup, find_packages

with open("README.rst", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="coral-nrel",
    author="Jake Nunemaker",
    description="Concurrent ORBIT for shared Resource Analysis Library",
    long_description=long_description,
    long_description_content_type="text/x-rst",
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
