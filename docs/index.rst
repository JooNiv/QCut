.. QCut documentation master file, created by
   sphinx-quickstart on Mon Jul 22 18:11:00 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QCut documentation
==================

QCut is a quantum circuit knitting package capable of efficiently partitioning
quantum circuits with wire and gate cuts using advanced LOCC and joint rotation decompositions
along with the standard local decompositions. It is designed to be compatible with Qiskit and should
be compatible with any Qiskit programmable backend but has been especially designed to be compatible
with IQM’s qpus and the Finnish Quantum Computing Infrastructure (`FiQCI <https://fiqci.fi/>`__).

QQCut has been built at CSC - IT Center for Science (Finnish IT Center for Science)

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Usage
   Options
   Examples
   Notebooks
   Theory
   Acknowledgement
   License
   QCut
   Changelog
   

Installation
------------

For installation a UNIX-like system is currently needed due to pymetis being used for automatic cut finding. On Windows use WSL

| **Pip:**
| Installation should be done via ``uv``

.. code:: bash

   uv pip install QCut
   #or
   uv add QCut

If using other than the default Qiskit version (newest) it is recommended to install Qiskit first before installing QCut.

Using uv is the recommended install method.

| **Install from source**
| It is also possible to use QCut by cloning this repository and
  including it in your project folder.

**IQM hardware and fake backends**

QCut ships an extra for this, so the adapter comes with it:

.. code:: bash

   uv pip install "QCut[iqm]"

That installs `IQM client <https://docs.meetiqm.com/iqm-client>`__ with its Qiskit
adapter, which covers Qiskit 1.0 up to but not including 2.2. Installing the extra will
therefore hold Qiskit below 2.2, put it in an environment whose Qiskit is already in
range if you would rather the resolver left it alone.

After installation just import the backend you want to use:

.. code:: python

   from iqm.qiskit_iqm import IQMFakeAdonis
   backend = IQMFakeAdonis()

