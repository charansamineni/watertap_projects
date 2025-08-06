*******************************
watertap_spacer_value
*******************************

.. image::
   https://github.com/charansamineni/watertap_spacer_value/workflows/Build%20Main/badge.svg
   :height: 30
   :target: https://github.com/charansamineni/watertap_spacer_value/actions
   :alt: Build Status

.. image::
   https://github.com/charansamineni/watertap_spacer_value/workflows/Documentation/badge.svg
   :height: 30
   :target: https://we3lab.github.io/watertap_spacer_value
   :alt: Documentation

.. image::
   https://codecov.io/gh/charansamineni/watertap_spacer_value/branch/main/graph/badge.svg
   :height: 30
   :target: https://codecov.io/gh/charansamineni/watertap_spacer_value
   :alt: Code Coverage

Modified RO unit models to handle custom spacer correlations

Features
========
- Store values and retain the prior value in memory
- ... some other functionality

Quick Start
===========
```python
from watertap_spacer_value import Example

a = Example()
a.get_value()  # 10
```

Installation
============
- **Stable Release:** `pip install watertap_spacer_value`
- **Development Head:** `pip install git+https://github.com/charansamineni/watertap_spacer_value.git`

Documentation
=============
For full package documentation please visit [charansamineni.github.io/watertap_spacer_value](https://charansamineni.github.io/watertap_spacer_value).

Development
===========

See [CONTRIBUTING.rst](CONTRIBUTING.rst) for information related to developing the code.

Useful Commands
===============

1. ``pip install -e .``

  This will install your package in editable mode.

2. ``pytest watertap_spacer_value/tests --cov=watertap_spacer_value --cov-report=html``

  Produces an HTML test coverage report for the entire project which can
  be found at ``htmlcov/index.html``.

3. ``docs/make html``

  This will generate an HTML version of the documentation which can be found
  at ``_build/html/index.html``.

4. ``flake8 watertap_spacer_value --count --verbose --show-source --statistics``

  This will lint the code and share all the style errors it finds.

5. ``black watertap_spacer_value``

  This will reformat the code according to strict style guidelines.

Legal Documents
===============
- `LICENSE <https://github.com/charansamineni/watertap_spacer_value/blob/main/LICENSE/>`_
