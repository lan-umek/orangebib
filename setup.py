#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orange3-Biblium
===============
Bibliometric analysis add-on for Orange data mining suite.
"""

from setuptools import setup, find_packages

NAME = "Orange3-Biblium"
VERSION = "0.1.0"
DESCRIPTION = "Bibliometric analysis widgets for Orange (powered by Biblium)"
LONG_DESCRIPTION = """
Orange3-Biblium (OrangeBib) provides widgets for bibliometric and scientometric 
analysis of academic literature. It integrates the Biblium library with Orange
data mining suite.

Supports data from major bibliographic databases:
- Scopus
- Web of Science
- OpenAlex
- PubMed
- Dimensions
- Lens.org
- And many more (30+ formats)

Features:
- Load data from multiple bibliographic database export formats
- Query OpenAlex API directly from Orange
- Predefined sample datasets for learning and testing
- Seamless integration with Orange's data analysis pipeline
"""

AUTHOR = "Lan Umek"
AUTHOR_EMAIL = ""
URL = "https://github.com/lan-umek/orange3-biblium"

KEYWORDS = [
    "orange3 add-on",
    "bibliometrics",
    "scientometrics",
    "citation analysis",
    "biblium",
    "openalex",
    "scopus",
    "web of science",
]

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: X11 Applications :: Qt",
    "Environment :: Plugins",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Visualization",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
]

PACKAGES = find_packages()

PACKAGE_DATA = {
    "orangebib": ["icons/*.svg"],
    "orangebib.widgets": ["icons/*.svg"],
}

INSTALL_REQUIRES = [
    "Orange3>=3.36.0",
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "requests>=2.28.0",
]

EXTRAS_REQUIRE = {
    "full": [
        "biblium>=2.12.0",  # Full Biblium library
    ],
}

ENTRY_POINTS = {
    "orange.widgets": [
        "Biblium = orangebib.widgets",
    ],
    "orange3.addon": [
        "Biblium = orangebib",
    ],
}

if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/plain",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        entry_points=ENTRY_POINTS,
        python_requires=">=3.9",
        zip_safe=False,
    )
