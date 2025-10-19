import numpy as np
import math
import scipy
import math
import networkx as nx
import matplotlib.pyplot as plt


def visualize_G(G):
    plt.figure(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, edge_color='gray')
    plt.show()


def X_S_generation(N, p):
    pX = np.exp(np.log(1 - p) * np.arange(N))
    pX = pX / sum(pX)
    X = np.random.multinomial(N - 1, pX)
    S = np.cumsum(X-1)
    return X, S

def X_S_positive_generation(N, p):
    pX = np.exp(np.log(1 - p) * np.arange(N))
    pX = pX / sum(pX)
    flag = 0
    while flag == 0:
        X = np.random.multinomial(N - 1, pX)
        S = np.cumsum(X-1)
        flag = (S[:-1] >= 0).all()  # & (S[-1] == -1)
    return X, S

def connected_gnp_generation(N, p, seed = None):

    G = nx.Graph()
    G.add_nodes_from(range(N))

    if not (seed is None):
        np.random.seed(seed)
        
    X, S = X_S_positive_generation(N, p)

    active_nodes = [0]
    inactive_nodes = set(range(1, N))

    for i in range(1, N):
        if not active_nodes:
            raise ValueError("No active nodes left.")
        active_node = active_nodes[0]
        num_edges = X[i - 1]
        neighbors = np.random.choice(list(inactive_nodes), num_edges, replace = False)

        for neighbor in neighbors:
            G.add_edge(active_node, neighbor)
            inactive_nodes.remove(neighbor)
            active_nodes.append(neighbor)
        for other_active in active_nodes:
            if other_active != active_node and np.random.random() < p:
                G.add_edge(active_node, other_active)
        active_nodes.remove(active_node)
        
    return G

def connected_gnp_generation_fast(N, p, seed = None):

    G = nx.Graph()
    G.add_nodes_from(range(N))

    if not (seed is None):
        np.random.seed(seed)
        
    X, S = X_S_positive_generation(N, p)
    P_set_size = np.sum(S[:-1])

    pos = 0
    indeces = []
    while True:
        pos += np.random.geometric(p)
        if pos > P_set_size:
            break
        indeces.append(pos)
    Ep = len(indeces)
 
    # random uniform permutation (starting from 0)
    perm = np.hstack([0, np.random.permutation(N-1)+1]) 
    # tree edges
    k = 0
    for t in range(N):
        i = perm[t]
        for j in perm[k+1 : k + 1+ X[t]]:
            G.add_edge(i, j)
        k += X[t]

    # P edges
    i = 0 
    t = 1 
    for j in range(Ep):
        while i + S[t-1]< indeces[j]:
            i += S[t-1]
            t += 1
        G.add_edge(perm[t], perm[t + indeces[j]-i])

    return G




def invert_f(x, tol=1e-12, max_iter=100):
    def f(c): return c * (1 + math.exp(-c)) / (1 - math.exp(-c))
    if x < 2:
        raise ValueError("x<2")
    if abs(x - 2) < tol:
        return 0.0
    l, r = 1e-15, 1.0
    while f(r) < x:
        r *= 2
    for _ in range(max_iter):
        m = 0.5 * (l + r)
        v = f(m)
        if abs(v - x) < tol:
            return m
        if v < x:
            l = m
        else:
            r = m
    raise ValueError("No convergence")


def connected_gnm_generation_fast(N, M, seed = None):

    G = nx.Graph()
    G.add_nodes_from(range(N))

    # p calculation
    if N <= M:
        c = invert_f(2 * M / (N - 1))
        p = c / N
    elif M == N-1:
        p = 0.00000001
    else:
        raise ValueError("M < N-1")

    if not (seed is None):
        np.random.seed(seed)
    
    flag = False
    while flag == False:
        X, S = X_S_positive_generation(N, p)
        P_set_size = np.sum(S[:-1])

        pos = 0
        indeces = []
        while True:
            pos += np.random.geometric(p)
            if pos > P_set_size:
                break
            indeces.append(pos)
        Ep = len(indeces)
        edges_num = N - 1 + Ep
        flag = (edges_num == M)
 
    # random uniform permutation (starting from 0)
    perm = np.hstack([0, np.random.permutation(N-1)+1]) 
    # tree edges
    k = 0
    for t in range(N):
        i = perm[t]
        for j in perm[k+1 : k + 1+ X[t]]:
            G.add_edge(i, j)
        k += X[t]

    # P edges
    i = 0 
    t = 1 
    for j in range(Ep):
        while i + S[t-1]< indeces[j]:
            i += S[t-1]
            t += 1
        G.add_edge(perm[t], perm[t + indeces[j]-i])

    return G

