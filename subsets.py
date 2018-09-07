from itertools import combinations
import numpy as np

def create_intial_conditions(number_of_species, number_of_condtions):

    def subsets(s):
        for cardinality in range(len(s) + 1):
            yield from combinations(s, cardinality)

    possible_elements = [ k+1 for k in range(number_of_species)]
    pre_subsets = [set(sub_set) for sub_set in subsets(possible_elements)]

    number_of_subsets = 2**(number_of_species)

    print(number_of_subsets)




    return type(pre_subsets[0])

print(create_intial_conditions(14))
