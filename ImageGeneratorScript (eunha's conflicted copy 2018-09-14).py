
from  SyntheticMicro import *
import numpy as np
from numpy import ndarray as Tensor
from scipy.integrate import solve_ivp


mu : float = 0.0
interspec : float= 0.6
nodes_number : int= 28
interaction_probability :float = 1
#np.random.seed(12)

A_matrix : Tensor = ER_Eco_Net_Matrix(nodes_number, interaction_probability, interspec )
# This matrix will be "broadcasted" to all Ax columns
r_matrix : Tensor = np.random.uniform(0,1, nodes_number)

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

def GetFinalState(eq_solution):
    return eq_solution.y[:, -1].tolist()

def GetImageMatrix(species_number:int,T:int) -> Tensor:
    sol_flag :int = 0
    states_container = []
    for k in range(species_number):
        while sol_flag == 0:
            initial_cond_ivp = Get_Random_Composition(species_number )
            ivp_sol = solve_ivp(SystemDynamics, (0,T), initial_cond_ivp, method='RK45')
            if ivp_sol.status != -1:
                sol_flag = 1
        sol_flag  = 0
        states_container.append(GetFinalState(ivp_sol))

    return np.array(states_container)


number_of_images : int = 100

T   = 75
images_dataset = [ ]

for k in range(number_of_images):
    images_dataset.append(GetImageMatrix(28, T))

import pickle


with open('images_dataset.pickle', 'rb') as outfile:
    pickle.dump(images_dataset, outfile)
