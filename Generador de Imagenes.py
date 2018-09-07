import networkx as nx
from itertools import combinations
import numpy as np
"""
Creamos la matriz A
"""
# Parametros

mu = 1
sigma = 0.6
nodes_number = 28
interaction_probability = 1

np.random.seed(12)

red = nx.erdos_renyi_graph(nodes_number, nodes_number, seed=12, directed=False)
adj_matrix = nx.to_numpy_matrix(red)



A_matrix = np.multiply(adj_matrix, (sigma * np.random.randn(28, 28) + mu) )
posibles = [1,2,3,4,5,6]

def findsubsets(s,m):
    return set(combinations(s, m))


my_sets = findsubsets(posibles,3)

#print(my_sets)


print(A_matrix)
