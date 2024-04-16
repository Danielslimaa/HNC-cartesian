import numpy as np 
import matplotlib.pyplot as plt
import math
import scipy 
from scipy import special
import scipy.special as sc
from sympy import *
import time
from time import strftime, gmtime
import numba
from numba import jit
import pandas as pd
import sys

import concurrent.futures
import multiprocessing
num_processes = multiprocessing.cpu_count()

from timeit import default_timer as timer

import sympy

font = {'family': 'serif',
          'color':  'black',
          'weight': 'normal',
          'size': 16,
          }

global first_term, second_term, third_term, V_ph, V_ph_k

N = 2*2048
k_max = N
L = 40

#del r_discretization, k_discretization
dl = L / N
l = np.linspace(-L,L,N)
X, Y = np.meshgrid(l,l)

g = np.ones((N,N), dtype=np.complex64)
S = np.ones((N,N), dtype=np.complex64)

#Initializing the terms
first_term = np.ones((N,N), dtype=np.complex64)
second_term = np.ones((N,N), dtype=np.complex64)
third_term = np.ones((N,N), dtype=np.complex64)

V_ph = np.ones((N,N), dtype=np.complex64)
V_ph_k = np.ones((N,N), dtype=np.complex64)