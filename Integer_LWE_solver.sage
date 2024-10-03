import numpy as np
import random
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
import time



def hamming_weight_vector(n, h):
    if h > n:
        raise ValueError("Hamming weight h cannot be greater than vector dimension n.")
    if h < 0:
        raise ValueError("Hamming weight h cannot be negative.")
    vector_data = [1] * h + [0] * (n - h)
    
    (random.shuffle(vector_data))
    hamming_vector = Matrix([vector_data])[0]
    
    return hamming_vector


def rand_vector(n,q): return vector([randint(-q,q) for i in range(n)])
def noise_vector(n): return vector([D() for i in range(n)])


def nonzero_positions(vector):
    positions = [index for index, value in enumerate(vector) if value != 0]
    return positions



def find_indices_of_largest_elements(lst):
    # Get sorted indices based on the values of the list in descending order
    sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x], reverse=True)
    return sorted_indices



def mod(a,q):
   temp = a%q
   if temp > (q-1)/2:
      temp -= q
   return temp


def vec_mod(v,q):
    temp = vector(ZZ, len(v))
    for i in range(len(v)):
        temp[i] = mod(v[i],q)
    return temp




def Integer_LWE_solver(n, Hw, q, m, sigma):
    D = DiscreteGaussianDistributionIntegerSampler(sigma=sigma)
    s = hamming_weight_vector(n, Hw)
    solindex = nonzero_positions(s)
#    print('sol=', solindex)
    
    Success_ratio_for_ours = 0
    Success_ratio_for_NMW = 0
    Success_ratio_for_CCP = 0
    
    for i in range(100):
        start = time.time()    
        A = matrix(ZZ, m, n)
        e = vector([D() for j in range(m)])
        for j in range(m):
            A[j] = rand_vector(n, (q - 1) / 2)
        b = A * s + e
        
        #### Ours
        Sol_Candidate_list = []
        At = A.transpose()
        for j in range(Hw):
            C = At * b
            max_value = max(list(C))
            Sol_Candidate_list += [list(C).index(max_value)]
            b -= At[Sol_Candidate_list[j]]
            At[Sol_Candidate_list[j]] = zero_vector(m)

        Sol_Candidate_list.sort()
        end = time.time()
        if Sol_Candidate_list == solindex:
            Success_ratio_for_ours += 1
        
        # #### NMW+24 analysis
        # Sol_Candidate_list_NMW = []
        # At = A.transpose()
        # for i in range(n):
        #     if std(b - At[i]) < std(b):
        #         Sol_Candidate_list_NMW += [i]

        # if Sol_Candidate_list_NMW == solindex:
        #     Success_ratio_for_NMW += 1
        
        #### CCP+24 analysis
        Sol_Candidate_list_CCP = []
        Threshold = round(sqrt(Hw) / sqrt(12))
        Mb = vector(ZZ, m)
        M = matrix(ZZ, m, n)
        pos = 0
        while pos < m:
            a = rand_vector(n, (q - 1) / 2)
            b = a * s + D()
            if b > Threshold * q:
                M[pos] = a
                Mb[pos] = b
                pos += 1

        Mt = M.transpose()
        Avg_set = [mean(Mt[j]) for j in range(n)]
        Sol_Candidate_list_CCP = find_indices_of_largest_elements(Avg_set)[:Hw]
        Sol_Candidate_list_CCP.sort()
        if Sol_Candidate_list_CCP == solindex:
            Success_ratio_for_CCP += 1

    return (#Success_ratio_for_ours / 100, 
            #Success_ratio_for_NMW / 100, 
            Success_ratio_for_CCP / 100
            #end - start
            )

