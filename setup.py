from setuptools import setup

setup(
    name="assoc_space",
    version='1.0.2',
    maintainer='Luminoso Technologies, Inc.',
    maintainer_email='dev@luminoso.com',
    license="MIT",
    url='http://github.com/LuminosoInsight/assoc-space',
    platforms=["any"],
    description="Computes association strength over semantic networks in a dimensionality-reduced form.",
    packages=['assoc_space'],
    install_requires=['numpy', 'scipy'],
)
