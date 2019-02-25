#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "kipoi>=0.6.0",
    "gin-config",
    "comet_ml",
    "numpy",
    "pandas",
    "tqdm",
    "colorlog",
    "argh",
    "scikit-learn",
    "fs",
    "fs-s3fs",
    # sometimes required
]

test_requirements = [
    "bumpversion",
    "wheel",
    "epc",
    "jedi",
    "pytest>=3.3.1",
    "pytest-xdist",  # running tests in parallel
    "pytest-pep8",
    "pytest-cov",
    "coveralls",
]

setup(
    name='gin-train',
    version='0.1.16',
    description="gin-train: model training boilerplate",
    author="Ziga Avsec",
    author_email='avsec@in.tum.de',
    url='https://github.com/avsecz/gin-train',
    long_description="gin-train: model training boilerplate",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": test_requirements,
    },
    entry_points={'console_scripts': ['gin_train = gin_train.__main__:main', 'gt = gin_train.__main__:main']},
    license="MIT license",
    zip_safe=False,
    keywords=["deep learning"],
    test_suite='tests',
    package_data={'gin_train': ['logging.conf']},
    tests_require=test_requirements
)

