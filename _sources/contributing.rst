Contributing
============

Thank you for considering contributing to cheese3d!

Development Setup
-----------------

1. Fork the repository
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/your-username/cheese3d.git
      cd cheese3d

3. Set up the development environment:

   .. code-block:: bash

      pixi install

4. Create a branch for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Submitting Changes
------------------

1. Make your changes work
2. Update documentation as needed
3. Commit your changes (with descriptive commit messages)
4. Push to your fork
5. Submit a pull request

Documentation
-------------

To build the documentation locally:

.. code-block:: bash

   # Using the predefined Pixi task
   pixi run docs

   # Or manually with the docs environment
   pixi run -e docs sphinx-build -b html docs/source docs/_build/html

To view the documentation in your browser:

.. code-block:: bash

   # Using the predefined Pixi task
   pixi run docs-serve

   # Or manually
   cd docs/_build/html && python -m http.server

The built documentation will be available at ``http://localhost:8000``.

Issues
------

Please report bugs and request features using the issue tracker.

When reporting bugs, please include:

* Description of the issue
* Steps to reproduce
* Expected behavior
* Actual behavior
* Environment details (OS, Python version, etc.)
