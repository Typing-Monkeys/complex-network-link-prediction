from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='complex-network-link-prediction',
    version='1.3',
    license='MIT',
    description='A python library for link prediction in social networks',
    author="Cristian Cosci, Fabrizio Fagiolo, Nicolò Vescera, Nicolò Posta, Tommaso Romani",
    packages=find_packages(where='.', include=['cnlp*']),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Typing-Monkeys/social-network-link-prediction',
    keywords='Link Prediction, Social Network, Complex Network Analisys',
    install_requires=[
        'networkx',
        'scipy',
        'numpy',
    ],
)
