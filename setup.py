#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Sophia Krix",
    author_email='sop3kri@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Using graph convolutional neural networks with multi-scale data on the MAVO knowledge graph for drug repositioning",
    entry_points={
        'console_scripts': [
            'multigml=multigml.cli:main',
        ],
    },
    install_requires=requirements,
    license="Apache license",
    long_description=readme + '\n\n',
    include_package_data=True,
    keywords='multigml',
    name='multigml',
    packages=find_packages(include=['MultiGML', 'MultiGML.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/SCAI-BIO/MultiGML',
    version='0.0.2-dev',
    zip_safe=False,
)
