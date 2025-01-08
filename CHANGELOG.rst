=========
Changelog
=========

Version 0.2.0
=============
- Single qubit cut gate now the default cut method
    * Greatly simplifies placing cuts.
    * Old two qubit gate deprecated and will be removed soon.
    * Check out documentation for migration help.

Version 0.1.3
=============
- Hotfix for source files not included in pypi build.
    * 0.1.0 - 0.1.2 not installable.

Version 0.1.2
=============
- Add Qiskit 1.0 support.
    * Supported versions now >= 0.45.3, < 1.2.
    * No worflow changes. No migration required.
    * Compatible with qiskit-iqm 13.15
- Add Python 3.11 support.
    * Supported versions now >= 3.9, < 3.12.
- Fix bug in :code:`_get_bounds()` method.
- Use :code:`pickle.loads(pickle.dumps())` instead of :code:`deepcopy()` in :code:`_get_experiment_circuits()`.
    * Slight performance improvement.
- Revert back to vx.x.x versioning scheme.


Version 0.1.1
=============

- Code quality improvements:
    * Move to pyproject.toml. Contents of ruff.toml and setup.py now live in pyproject.toml.
    * Relative imports are now absolute imports.
    * All images are now located under the _static folder.
- Added Github workflows:
    * Added github actions workflows for building the documentation and testing that the documentation can be built.
    * Added github actions workflows for testing the code.
    * Added github actions workflows for publishing to pypi.
    * Added github actions workflows for linting.
- Revamped documentation:
    * Documentation now uses the Book theme.
    * Fixed all warnings from sphinx when building the documentation.
    * Fixed spelling mistakes.
    * Added a new page for the changelog.
- README now contains the information to build the docs.
- Change versioning scheme to "x.x.x" instead of "vx.x.x".

Version 0.1.0
=============

- First stable release
