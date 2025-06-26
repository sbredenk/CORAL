# Contributor's Guide

(contributing:getting-started)=
## Getting Started

These contributing guidelines should be read by software developers wishing to contribute code or
documentation changes into CORAL, or to push changes upstream to the main NREL/CORAL repository.

1. Create a fork of CORAL on GitHub
2. Clone your fork of the repository

    ```bash
    git clone -b develop https://github.com/<your-GitHub-username>/CORAL.git
    ```


## Keeping your fork in sync with NREL/CORAL

The "main" CORAL repository is regularly updated with ongoing research at NREL. After
creating and cloning your fork from the previous section, you might be wondering how to keep it
up to date with the latest improvements.

Please note that the below process may introduce merge conflicts with your work, and this does not
provide guidance about how to deal with those conflicts. Here is a good resource for working on
[merge conflicts](https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts) that will
inevitably arise in development work.

1. Ensure you're in the CORAL folder. This may look different depending on your operating system.

   ```bash
   cd /your/path/to/CORAL/
   ```

2. If you haven't already, add NREL/CORAL as the "upstream" location (or whichever naming
   convention you prefer).

   ```bash
   git remote add upstream https://github.com/NREL/CORAL.git
   ```

   To find the name you've given NREL/CORAL again, you can simply run the following to display
   all the remote sources you're tracking.

   ```bash
   git remote -v
   ```

3. Fetch all the remote changes

   ```bash
   git fetch --all
   ```

4. Sync the upstream changes

   ```bash
   # If there was a new release this will need to be updated
   git checkout main
   git pull upstream main

   # Most common branch to bring up to speed
   git checkout develop
   git pull upstream develop
   ```

5. Bring your feature branch up to date with the latest changes, assuming you started from the
   develop branch.

   ```bash
   git checkout feature/your_contribution
   git merge develop
   ```

## Issue Tracking

New feature requests, changes, enhancements, non-methodology features, and bug reports can be filed
as new issues in the [Github.com issue tracker](https://github.com/NREL/CORAL/issues) at any time.
Please be sure to fully describe the issue.

For other issues, please email sophie.bredenkamp@nrel.gov.

### Issue Submission Checklist

1. Does the issue already exist?
   Yes: If you find your issue already exists, make relevant comments and add your
   [reaction](https://github.com/blog/2119-add-reactions-to-pull-requests-issues-and-comments).
   Use a reaction in place of a "+1" comment:

   - 👍 - upvote
   - 👎 - downvote

2. Is this an individual bug report or feature request?
3. Can the bug be easily reproduced?
   1. Be sure to include enough details about your setup and the issue you've encountered
   2. Simplify as much of the code as possible to better isolate the problem
4. Will someone else understand the issue or change requested given the information provided?

## Repository

The CORAL repository is hosted on Github, and located here: http://github.com/NREL/CORAL

This repository is organized using a modified git-flow system. Branches are organized as follows:

- main: Stable release version. Must have good test coverage and may not have all the newest features.
- develop: Development branch which contains the newest features. Tests must pass, but code may be
  unstable.
- feature/xxx: Feature ranch from develop, should reference a GitHub issue number.
- fix/xxx: Bug fix branch from develop, should reference a GitHub issue number. Can be based off
  main if this is a necessary patch.

To work on a feature, please fork CORAL first and then create a feature branch in your own fork.
Work out of this feature branch before submitting a pull request.

Be sure to periodically synchronize the upstream develop branch into your feature branch to avoid
conflicts in the pull request.

When your branch is ready, make a pull request to NREL/CORAL through the
[GitHub web interface](https://github.com/NREL/CORALpulls).


## Documentation

Documentation is written primarily using Markdown, with some components written in
ReStructured Text, and is located in the `CORAL/docs/` directory. Additionally, all method and class
documentation is written as NumPy-style docstrings in the code itself, with some aspects documented
inline as needed.

If the `docs` extras haven't already been installed, be sure to do so before you attempt to build
the documentation site.

```bash
# Build the docs
jupyter-book build docs
```


(contributing:pull-request)=
## Pull Request

Pull requests must be made for all changes. Most pull requests should be made against the develop
branch unless patching a bug that needs to be addressed immediately, and only core developers should
make pull requests to the main branch.

All pull requests, regardless of the base branch, must include updated documentation. In addition, 
code coverage should not be significantly negatively affected.

### Scope

Encapsulate the changes of one issue, or multiple if they are highly related. Three small pull
requests is greatly preferred over one large pull request. Not only will the review process be
shorter, but the review will be more focused and of higher quality, benefitting the author and code
base. Be sure to write a complete description of these changes in the pull request body.


### Documentation

Include any relevant changes to inline documentation, docstrings, and any of the documentation files
located in `CORAL/docs/`. Pull requests will not be accepted until these changes are complete.

Be sure to build the documentation prior to submission.

### Changelog

All changes must be documented appropriately in CHANGELOG.md in the [Unreleased] section.

## Release Process

This section is a reference for CORAL's maintainers to keep processes largely consistent
over time, regardless of who the core developers are.

1. Bump version number and metadata in `CORAL/__init__.py`
2. Bump version numbers of any dependencies in `setup.py`. Be sure to separate to keep dependencies
   separated by what they are required for (see the `project.optional-dependencies` section of
   `pyproject.toml`)
3. Update the changelog at `CORAL/CHANGELOG.md`, changing the "UNRELEASED" section to the new
   version and the release date (e.g. "[2.3 - 2022-01-18]").
4. Make a pull request into develop with these updates, and be sure to follow the guide in
   [Pull Requests](contributing:pull-request).

5. Merge develop into main through the git command line

  ```bash
   git checkout main
   git merge develop
   git push
   ```
