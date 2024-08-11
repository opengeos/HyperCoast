# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/opengeos/HyperCoast/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with `bug` and
`help wanted` is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
`enhancement` and `help wanted` is open to whoever wants to implement it.

### Write Documentation

HyperCoast could always use more documentation,
whether as part of the official HyperCoast docs,
in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/opengeos/HyperCoast/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to implement.
-   Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up HyperCoast for local development.

1.  Fork the HyperCoast repo on GitHub.

2.  Clone your fork locally:

    ```shell
    $ git clone https://github.com/<YOUR-GITHUB-USERNAME>/HyperCoast.git
    ```

3.  Create a new conda environment to install HyperCoast and its dependencies. Assuming you have
    [Anaconda](https://www.anaconda.com/distribution/#download-section) or
    [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed,
    this is how you set up your fork for local development:

    ```shell
    $ conda install -n base mamba -c conda-forge
    $ conda create -n hyper python=3.11
    $ conda activate hyper
    $ mamba install -c conda-forge hypercoast cartopy earthaccess mapclassify pyvista
    $ cd HyperCoast/
    $ pip install -e .
    ```

4.  Create a branch for local development:

    ```shell
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

5.  When you're done making changes, check that your changes pass
    [pre-commit](https://pre-commit.com/) and the tests:

    ```shell
    $ pip install pre-commit
    $ pre-commit install
    $ pre-commit run --all-files
    $ python -m unittest discover tests/
    ```

6.  Commit your changes and push your branch to GitHub:

    ```shell
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

7.  Submit a pull request through the GitHub website.
8.  Check the status of your pull request on GitHub and make sure
    that the tests for the pull request pass for all supported Python versions.
9.  Commit more changes to your branch to fix the text errors if necessary.
10. Wait for the pull request to be reviewed by the maintainers.
11. Congratulations! You've made your contribution to HyperCoast!

## Contributor Agreements

Before your contribution can be accepted, you will need to sign the appropriate contributor agreement. The Contributor License Agreement (CLA) assistant will walk you through the process of signing the CLA. Please follow the instructions provided by the assistant on the pull request.

-   [Individual Contributor Exclusive License Agreement](https://github.com/opengeos/HyperCoast/blob/main/docs/cla.md)
-   [Entity Contributor Exclusive License Agreement](https://github.com/opengeos/HyperCoast/blob/main/docs/cla.md)
