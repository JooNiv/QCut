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

Wall clock in this suite is almost entirely simulated shots: a cut experiment enumerates
one circuit per combination of QPD terms, so the count multiplies with every cut, and
each circuit then wants thousands of shots. Qiskit already spreads a single simulation
across every core, so running the tests in parallel does not help. Three tiers are
marked for that reason:

```shell
pytest -m "not sim"     # ~5s, no simulation at all: QPD tables, gammas, planning
pytest -m "not slow"    # ~95s, everything but the heaviest end-to-end cases
pytest                  # ~230s, the lot
```

Use the first while working, the last before pushing. New tests join the quick tier
automatically; add `@pytest.mark.sim` to anything that puts a circuit on a simulator,
and `@pytest.mark.slow` as well if it costs more than about ten seconds.

Keep an eye on shot budgets in new tests. Most of these assert that a cut circuit
reconstructs its original, and that fails by order one when it fails at all, so a few
thousand shots is usually plenty. If you do reduce a budget, measure the worst error
over several runs before choosing the tolerance rather than reasoning about it: a
tolerance with only a few percent of headroom is a flaky test waiting to happen.

The pytest tests will be ran against multiple Python versions using Github actions when you make a pull request.

If your changes should retain backwards compatibility to older Qiskit versions additionally run:

```shell
tox --parallel
```

Finally make sure that documentation is up to date and builds properly by running:

```shell
uv sync --group docs
cd docs
uv run sphinx-build -v -b html . build/sphinx/html -W
```
