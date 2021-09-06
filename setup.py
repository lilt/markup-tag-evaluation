import setuptools
from markup_tag_evaluation import __version__

dev_packages = ["mypy>=0.79", "pycodestyle>=2.6"]
scripts = ["scripts/evaluate-markup-tags.py"]


setuptools.setup(
    name="markup-tag-evaluation",
    version=__version__,
    author="Thomas Zenkel",
    author_email="thomas@lilt.com",
    packages=["markup_tag_evaluation"],
    scripts=scripts,
    extras_require={"dev": dev_packages},
    python_requires=">=3.8",
)
