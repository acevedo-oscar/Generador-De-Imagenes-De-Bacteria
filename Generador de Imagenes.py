import networkx as nx
from itertools import combinations
import numpy as np

# Analogo de Subsets de wolfram
def findsubsets(s,m):
    return set(combinations(s, m))

"""
Creamos la matriz A
"""
# Parametros

mu = 1
sigma = 0.6
nodes_number = 28
interaction_probability = 1

np.random.seed(12)

# Step 1
red = nx.erdos_renyi_graph(nodes_number, nodes_number, seed=12, directed=False)
adj_matrix = nx.to_numpy_matrix(red)

random_normal_matrix = sigma * np.random.randn(nodes_number, nodes_number) + mu
A_half = np.multiply(adj_matrix, random_normal_matrix )
A_matrix = np.subtract(A_half, np.identity(nodes_number))

# Step 2

# This matrix will be "broadcasted" to all Ax columns
r_matrix = np.random.uniform(0,1, nodes_number)



posibles = [1,2,3,4,5,6]
my_sets = findsubsets(posibles,3)

"""

print(my_sets)
print(A_matrix.shape)
print(A_matrix[0:4,0:4])

"""
