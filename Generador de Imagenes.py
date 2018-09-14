import networkx as nx
import numpy as np

from numpy import ndarray as Tensor

from scipy.integrate import solve_ivp

def Get_Random_Composition(species_n : int  ) -> Tensor:


    rand_int : int = np.random.randint(1,2**species_n)
    binary_digit_list = [int(x) for x in str(format(rand_int, "b"))]
    while len(binary_digit_list) != species_n:
        binary_digit_list.append(0)


    random_uniform_vector: Tensor = np.random.uniform(0,1,(1, species_n) )

    rand_comp = np.multiply(np.array(binary_digit_list), random_uniform_vector )


    return rand_comp.tolist()[0]

def Diagonal_Matrix(input_vector : Tensor ) -> Tensor :
     return np.multiply(np.identity(input_vector.shape[0]),input_vector)

def ER_Eco_Net_Matrix(species_number : int, interaction_probability:float, interspecies_interaction: float ) -> Tensor:

    delta : float = 1.0
    mu : float = 0

    raw_net = nx.erdos_renyi_graph(species_number, species_number, seed=12, directed=False)
    adj_matrix :Tensor = nx.to_numpy_matrix(raw_net)

    # Reminder np.random.randn(m,n) returns a matrix of size (m,n)
    random_normal_matrix : Tensor = interspecies_interaction * np.random.randn(species_number, species_number) + mu
    A_half :Tensor = np.multiply(adj_matrix, random_normal_matrix )
    Full_matrix : Tensor = np.subtract(A_half, delta*np.identity(species_number))

    return Full_matrix



"""
Creamos la matriz A
"""
# Parametros

mu : float = 1.0
interspec : float= 0.6
nodes_number : int= 28
interaction_probability :float = 1

#np.random.seed(12)

# Step 1

A_matrix : Tensor = ER_Eco_Net_Matrix(nodes_number, interaction_probability, interspec )


# Step 2

# This matrix will be "broadcasted" to all Ax columns
r_matrix : Tensor = np.random.uniform(0,1, nodes_number)

#Step 3

time_interval = np.linspace(0,600,1000)

def SystemDynamics(  time , input_values ):
    """
    dx/dt = diag(x)[Ax+r]
    """
    A_m = A_matrix
    r_m = r_matrix
    diag_m = Diagonal_Matrix( np.array(input_values))

    left_product = np.matmul(A_m, np.array(input_values) ) + r_m

    dx =  np.matmul(diag_m,left_product.T).flatten()

    return dx.tolist()[0]

initial_cond = Get_Random_Composition(nodes_number )

T : int = 600
solution1 = solve_ivp(SystemDynamics, (0,T), initial_cond  )

print(solution1.message)

initial_cond2 = Get_Random_Composition(nodes_number )

solution2 = solve_ivp(SystemDynamics, (0,T), initial_cond2  )
print(solution2.message)


print(initial_cond)
print(initial_cond2)


"""

print(my_sets)
print(A_matrix.shape)
print(A_matrix[0:4,0:4])

"""
