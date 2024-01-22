CORAL
=====

Concurrent ORBIT for shared Resource Analysis Library


:Authors: `Jake Nunemaker <https://www.linkedin.com/in/jake-nunemaker/>`_, `Matt Shields <https://www.linkedin.com/in/matt-shields-834a6b66/>`_, `Sophie Bredenkamp <https://www.linkedin.com/in/sophie-bredenkamp-839321150/>`_


Development Setup
-----------------


Environment
~~~~~~~~~~~

A couple of notes before you get started:
 - It is assumed that you will be using the terminal on MacOS/Linux or the
   Anaconda Prompt on Windows. The instructions refer to both as the
   "terminal", and unless otherwise noted the commands will be the same.
 - To verify git is installed, run ``git --version`` in the terminal. If an error
   occurs, install git using these `directions <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.
 - The listed installation process is intended to be the easiest for any OS
   to get started. An alternative setup that doesn't rely on Anaconda for
   setting up an environment can be followed
   `here <https://realpython.com/python-virtual-environments-a-primer/#managing-virtual-environments-with-virtualenvwrapper>`_.

Instructions
~~~~~~~~~~~~

1. Download the latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
   for the appropriate OS. Follow the remaining `steps <https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
   for the appropriate OS version.
2. From the terminal, install pip by running: ``conda install -c anaconda pip``
3. Next, create a new environment for the project with the following.

    .. code-block:: console

        conda create -n <environment_name> python=3.7 --no-default-packages

   To activate/deactivate the environment, use the following commands.

    .. code-block:: console

        conda activate <environment_name>
        conda deactivate <environment_name>

4. Clone the repository:
   ``git clone https://github.com/NREL/CORAL.git``
5. Navigate to the top level of the repository
   (``<path-to-CORAL>/CORAL/``) and install ORBIT as an editable package
   with following command.

    .. code-block:: console

       # Note the "." at the end
       pip install -e .

       # OR if you are you going to be contributing to the code or building documentation
       pip install -e '.[dev]'
6. Install ORBIT wherever you would like (in coral path works: ``<path-to-CORAL>/``) using the following command.
	
	pip install orbit-nrel
	

Dependencies
~~~~~~~~~~~~

- Python 3.7+
- marmot-agents
- NumPy
- SciPy
- Matplotlib
- OpenMDAO (>=3.2)

Development Specific
~~~~~~~~~~~~~~~~~~~~

- black
- isort
- pre-commit
- pytest
- sphinx
- sphinx-rtd-theme


Recommended packages for easy iteration and running of code:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- jupyterlab
- pandas


Setting Up a Run
----------------
See coral_example_setup.ipynb for detailed explanation on running the model and understanding outputs.

