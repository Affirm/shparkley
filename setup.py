import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'future',
    'mock',
    'pyspark',
]

setuptools.setup(
    name="shparkley",
    version="0.0.3",
    install_requires=REQUIRED_PACKAGES,
    author="niloygupta",
    author_email="niloy.gupta@affirm.com",
    description="Scaling Shapley Value computation using Spark",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/Affirm/shparkley",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
