"""Marks ``tests`` as a regular package so ``from tests.parity.utils import ...``
resolves to this directory rather than any same-named package installed in
site-packages (a namespace-package directory loses to a regular package found
later on ``sys.path``).
"""
