import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="margingame",
    version="0.3.2",
    author="WillDudley",
    author_email="Will2346@live.co.uk",
    description="A package for the margin game.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WillDudley/margingame",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)