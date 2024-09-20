=========
Changelog
=========

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
