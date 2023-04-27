from setuptools import setup, find_packages

setup(
    name='complex-network-link-prediction',
    version='0.1',
    license='MIT',
    author=
    "Cristian Cosci, Fabrizio Fagiolo, Nicolò Vescera, Nicolò Posta, Tommaso Romani",
    packages=find_packages('cnlp'),
    package_dir={'': 'cnlp'},
    url='https://github.com/Typing-Monkeys/social-network-link-prediction',
    keywords='Link Prediction, Social Network, Complex Network Analisys',
    install_requires=[
        'networkx',
        'scipy',
        'numpy',
    ],
)
