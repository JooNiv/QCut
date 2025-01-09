<h1>Contributing to QCut</h1>

<h2>Setting up your environment</h2>
QCut is written in Python so you will need Python installed. Currently supported Python versions are >=3.9, <3.12.

Once Python is installed create a virtual environment for development:
```shell
python3 -m venv ~/.venvs/qcut-dev
```

Now to activate the environment run the script corresponding to your system:

<h3>Windows:</h3>

```shell
# In cmd.exe
~/.venvs/qcut-dev\Scripts\activate.bat
# In PowerShell
~/.venvs/qcut-dev\Scripts\Activate.ps1
```

<h3>Linux and MacOS:</h3>

```shell
source  ~/.venvs/qcut-dev/bin/activate
```

<h2>Install QCut dependencies</h2>

Install the Python packages, that QCut depends on, in your Python virtual environment.

```shell
pip install qiskit==1.1.2 qiskit-experiments==0.7.0 numpy qiskit-aer==0.13.3
```

Additionally install some development dependencies

```shell
pip install pytest ruff sphinx
```

<h2>Create a fork of QCut</h2>

In GitHub navigate to the QCut repository and click on fork. Once you have your fork of QCut clone it to your local development folder.
Now all that is left to do is to create a new branch and start working. Once you have made your changes create a pull request to merge your changes to QCut.
Before creating a pull request make sure that all the tests are passing:

```shell
python -m pytest
```

then lint all files to make sure you are following the proper style:

```shell
ruff check --fix
```

after running ruff fix any remaining issues.

Also make sure that documentation is up to date and builds properly by navigating to QCut/docs/ and running:

```shell
make clean
make html
```
