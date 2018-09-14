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
