# **Complex Network Link Prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- [![PyPi](https://badge.fury.io/py/nomepacchetto.svg)](https://badge.fury.io/py/nomepacchetto) --> <!-- [![Downloads](https://pepy.tech/badge/nomepacchetto/month)](https://pepy.tech/project/nomepacchetto) --> [![Wiki](https://img.shields.io/badge/howTo-Wiki-blue.svg)](https://github.com/Typing-Monkeys/social-network-link-prediction/wiki) [![GitHubIssues](https://img.shields.io/badge/issue_tracking-github-blue.svg)](https://github.com/Typing-Monkeys/social-network-link-prediction/issues) [![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)


#### **Complex Network Link Prediction** is a python library that implements some of the main techniques and algorithms to perform link predictions.


Check out our [home page](docs link)link pages qui for more information.

<img src="https://raw.githubusercontent.com/Typing-Monkeys/social-network-link-prediction/develop/imgs/logo.png" alt="logo" width="70%" />

This library implemented in python allows you to use some of the main algorithms and methods to perform link predictions. It was designed to carry out these tasks in **Complex Networks** and, specifically, in **Social Networks**. Each method has its own specific documentation available on the following page ---- where it is possible to see the required parameters and the output of the method itself. <br>
The methods are distinguished by belonging to categories and subcategories, below is an example image with all the categories.

<img src="https://raw.githubusercontent.com/Typing-Monkeys/social-network-link-prediction/develop/imgs/methods_list.jpg" alt="methods list" width="50%" />

The speed of computation differs both from the type of method and from the input graph. However, for convention and efficiency we have chosen to use the `csr_matrix` sparse matrix structure from the ***scipy*** library in each algorithm.


## Install
```
pip install cnlp
```

<hr>


## How to use
```python
import networkx as nx
import matplotlib.pyplot as plt
from cnlp.utils import to_adjacency_matrix
from cnlp.probabilistich_methods import stochastic_block_model

G = nx.karate_club_graph()
A = to_adjacency_matrix(G)

res = stochastic_block_model(G, 10)

predicted_edges = []
for u in range(res.shape[0]):
    for v in range(u + 1, res.shape[1]):
        if G.has_edge(u, v):
            continue
        w = res[u, v]
        predicted_edges.append((u, v, w))

# Sort the predicted edges by weight in descending order
predicted_edges.sort(key=lambda x: x[2], reverse=True)

# Print the predicted edges with their weight score
print("Top Predicted edges:")
for edge in predicted_edges[:50]:
    print(f"({edge[0]}, {edge[1]}): {edge[2]}")

nx.draw(G)
plt.show()
```

<hr>

### Contribute 💥
As there are still many methods to implement and, at the same time, maintaining a library takes up a lot of time, we are always happy to accept new willing and able people to contribute and support our project.

If you want to contribute or you've found a bug, you can open a **Pull Request**.

Check this [tutorial](#https://github.com/Typing-Monkeys/social-network-link-prediction/wiki/monkeflow-Workflow-🦍) if you want to use our preferred *Workflow* 🦍 for developing.

Otherwise you can open a normal Pull Request using `git` and help us to make this project even better!

### Help ❓
If you encounter any bug or you have some problem with this package you can open an [issue](#https://github.com/Typing-Monkeys/social-network-link-prediction/wiki/monkeflow-Workflow-🦍) to report it, we will resolve that asap.

<hr>

### Building From Source
???

### Dependencies
If your system does not have some or all of this requirements they will be installed during the istallation of this library
- networkx
- scipy
- numpy
ecc

<hr>

## References

- ***Ajay Kumar, Shashank Sheshar Singh, Kuldeep Singh, Bhaskar Biswas. 2020.***
    [Link prediction techniques, applications, and performance: A survey](https://www.sciencedirect.com/science/article/pii/S0378437120300856)
    *Physica A: Statistical Mechanics and its Applications*,
    ISSN 0378-4371, https://doi.org/10.1016/j.physa.2020.124289.
- ***David Liben-Nowell and Jon Kleinberg. 2003.***
    [The link prediction problem for social networks](https://dl.acm.org/doi/10.1145/956863.956972)
    *In Proceedings of the twelfth international conference on Information and knowledge management (CIKM '03). Association for Computing Machinery, New York, NY, USA*, 556–559, https://doi.org/10.1145/956863.956972.
- ***Víctor Martínez, Fernando Berzal, and Juan-Carlos Cubero. 2016.***
    [A Survey of Link Prediction in Complex Networks](https://dl.acm.org/doi/10.1145/3012704)
    *ACM Comput. Surv. 49, 4, Article 69 (December 2017), 33 pages*. https://doi.org/10.1145/3012704

<hr>

### ***Authors***

| ![cosci](https://avatars.githubusercontent.com/u/44636000?s=421&v=4) | ![vescera](https://avatars.githubusercontent.com/u/10250769?s=421&v=4)| ![fagiolo](https://avatars.githubusercontent.com/u/44865237?v=4) | ![romani](https://avatars.githubusercontent.com/u/44830726?v=4)| ![posta](https://avatars.githubusercontent.com/u/44830740?v=4) 
| - | - | - | - | - |
| [Cristian Cosci](https://github.com/CristianCosci) 🐔 | [Nicolò Vescera](https://github.com/ncvescera) 🦧 | [Fabrizio Fagiolo](https://github.com/F-a-b-r-i-z-i-o) 🐛 |  [Tommaso Romani](https://github.com/TommasoRomani) 🦍 | [Nicolò Posta](https://github.com/NicoloPosta) 🐒

