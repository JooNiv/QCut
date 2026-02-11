<h1>Contributing to QCut</h1>

<h2>Create a fork of QCut</h2>

In GitHub navigate to the QCut repository and click on fork. Once you have your fork of QCut clone it to your local machine.

<h2>Prepare environment</h2>

Using uv initialise the development environment:
```shell
uv sync
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

<h2>Working on you changes</h2>

Create a new branch for your changes. Once you are done lint and type check all files to make sure you are following the proper style:

```shell
ruff check --fix
uvx ty check /QCut
```

after running ruff and ty fix any remaining issues.

Also make sure that all tests pass:

```shell
pytest --cov
```

The pytest tests will be ran againts multiple Python versions using Github actions when you make a pull request.

If your changes should retain backwards compatibility to older Qiskit versions additionally run:

```shell
tox --parallel
```

Finally make sure that documentation is up to date and builds properly by navigating to QCut/docs/ and running:

```shell
uv pip install -r requirements-docs.txt
sphinx-build -v -b html . build/sphinx/html -W
```
