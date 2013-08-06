from setuptools import setup

setup(
    name="assoc_space",
    version='0.1',
    maintainer='Luminoso Technologies, Inc.',
    maintainer_email='dev@luminoso.com',
    license="MIT",
    url='http://github.com/LuminosoInsight/assoc-space',
    platforms=["any"],
    description="Computes association strength over semantic networks in a dimensionality-reduced form.",
    packages=['assoc_space'],

    # We also require numpy and scipy, but don't let pip know that or else it will try to compile them.
    install_requires=['ordered_set'],
)
