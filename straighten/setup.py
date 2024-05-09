from pathlib import Path

from setuptools import setup, find_packages

name = 'straighten'
classifiers = '''Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10'''

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

# get the current version
with open(Path(__file__).resolve().parent / name / '__version__.py', encoding='utf-8') as file:
    scope = {}
    exec(file.read(), scope)
    __version__ = scope['__version__']

setup(
    name=name,
    packages=find_packages(include=(name,)),
    include_package_data=True,
    version=__version__,
    descriprion='Interpolation along multidimensional curves',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/neuro-ml/straighten',
    download_url='https://github.com/neuro-ml/straighten/archive/v%s.tar.gz' % __version__,
    keywords=['differential geometry', 'basis', 'curves'],
    classifiers=classifiers.splitlines(),
    install_requires=requirements,
)
