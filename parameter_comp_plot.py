import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from bc_algorithms import ackley, rastrigin, sgd, cbo, naive, naive_uniform
from variant_1 import component_wise_diff
from variant_2 import common_noise
from variant_3 import random_mini_batch
from variant_1and3 import comp_diff_random_batch

def accuracy():
    pass

def tradeoff(ax, f, N=100, M=10, n=10000, m=50, lr=0.1, sigma=np.sqrt(0.1), lam=1):
    pass
