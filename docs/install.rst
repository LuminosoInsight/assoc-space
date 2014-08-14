Installing assoc-space
======================

assoc-space is designed to run on Python 2.7, 3.3, or 3.4.

The easiest way to install assoc-space is to use the ``pip`` package installer::

    pip install assoc-space

assoc-space depends on the NumPy and SciPy libraries. ``pip`` will try to
install them automatically, but this won't work on some systems.

As with most Python packages, it is best to install assoc-space in a virtualenv_,
not in a system-wide Python environment.

.. _virtualenv: http://docs.python-guide.org/en/latest/dev/virtualenvs/


Possible installation problems
------------------------------

Problem: I don't have a ``pip`` command.
    - Consider upgrading to Python 3.4, which has ``pip`` built in.
    - If you have the older ``easy_install`` but not ``pip``, you can first run ``easy_install pip``.
    - Otherwise, go follow the instructions for `installing pip`_.

.. _`installing pip`: http://pip.readthedocs.org/en/latest/installing.html


Problem: ``pip`` says I don't have permission to install things.
    Use virtualenv_, and then you will have a local Python environment that
    you can do whatever you want to.


Problem: The installation process for NumPy or SciPy crashes.
    Possible solutions:

    - Get the dependencies you need for building SciPy. On Ubuntu, one way
      to do that is::

        sudo apt-get build-dep scipy

    - If you can install SciPy system-wide but can't do it inside of a
      virtualenv_, one appreach you can take is to install virtualenvwrapper_
      and run the ``toggleglobalsitepackages`` command. (There are also more
      specific solutions, such as running ``add2virtualenv`` on your NumPy
      and SciPy installations.)

    - If you can't install SciPy at all, consider getting a distribution of
      Python that already has it installed, such as Anaconda_.

.. _virtualenvwrapper: http://virtualenvwrapper.readthedocs.org/en/latest/
.. _Anaconda: http://continuum.io/downloads#34


Problem: I'm on a Mac and something weird is happening involving ``/System/Frameworks``.
    Even though Mac OS comes with Python, you will probably be
    better off installing a Python development environment using Homebrew_.
    Once Homebrew is installed, you can install an up-to-date Python with
    this command, for example::

        brew install python3

.. _Homebrew: http://brew.sh
