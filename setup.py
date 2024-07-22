"""Setup."""

from setuptools import find_packages, setup

setup(
    name="QCut",
    version="0.0.4",
    author="Joonas Nivala",
    author_email="joonas.nivala@gmail.com",
    description="""A package for performing wire cuts of hardware without reset-gates or mid-circuit measurements.Built on top of qiskit""",  # noqa: E501
    long_description=open("README.md").read(),  # noqa: PTH123, SIM115
    long_description_content_type="text/markdown",
    url="https://github.com/JooNiv/CircuitKnitting",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9, <3.11",
    install_requires = [
        "qiskit == 0.45.3",
        "numpy < 2.0.0",
        "qiskit_aer == 0.13.3",
        "qiskit_experiments == 0.7.0",
    ],
)
