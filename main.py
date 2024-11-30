########################## ---- IMPORT STUFF ---- ##########################
############################################################################

import io, os, sys, math, cmath, time, heapq
from collections import *
from random import randint

# sys.setrecursionlimit(10**6)
start_time = time.time()
min_value = -float('inf')
max_value = float('inf')
RANDOM = randint(1, 10**9)
mod = 10**9+7
# mod = 998244353

d4 = [[0, -1], [0, 1], [-1, 0], [1, 0]]
d8 = [[0, -1], [0, 1], [-1, 0], [1, 0], [1, -1], [1, 1], [-1, -1], [-1, 1]]

###################### ---- HANDLING COLLISIONS ---- #######################
############################################################################

class Wrapper(int):
    def __init__(self, x):
        int.__init__(x)

    def __hash__(self):
        return super(Wrapper, self).__hash__() ^ RANDOM

########################### ---- MAIN CODE ---- ############################
############################################################################

def code():
    

def main():
    t = 1
    # t = inp()
    for tt in range(t):
        code()
    
    if not ONLINE_JUDGE:
        sys.stdout.write("\nTime: "+f'{((time.time()-start_time)*1000):.3f}'+" ms")
        sys.stdout.close()

################### ---- USER DEFINED I/O FUNCTIONS ---- ###################
############################################################################

def inp():
    return(int(input()))
def inlt():
    return(list(map(int,input().split())))
def invr():
    return(map(int,input().split()))
def insr():
    return(input().strip().decode('utf-8') if ONLINE_JUDGE else input().strip())
def bnsr():
    return([i.decode('utf-8') if ONLINE_JUDGE else i for i in input().strip().split()])
def opt(n):
    sys.stdout.write(str(n)+" ")
def opln(n):
    sys.stdout.write(str(n)+"\n")
def oplt(list):
    sys.stdout.write(" ".join(map(str, list))+"\n")

########################## ---- JUDGE SYSTEM ---- ##########################
############################################################################

ONLINE_JUDGE = not os.path.exists('opt.txt')

if ONLINE_JUDGE:
    input = io.BytesIO(os.read(0,os.fstat(0).st_size)).readline
    # input = sys.stdin.readline
else:
    sys.stdin = open("inp.txt", "r")
    sys.stdout = open("opt.txt", "w")

main()

########################### ---- ENDS HERE ---- ############################
############################################################################