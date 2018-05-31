import networkx as nx
import networkx.generators.stochastic as rg
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

n = 10
p = 0.5

# G = rg.gnp_random_graph(n, p)
# G=nx.complete_graph(10)
G = nx.erdos_renyi_graph(n, p)

s = nx.is_connected(G)
print("connected:", s)

for i in range(5):
    G.add_edge(i, i)

G = nx.to_undirected(G)
A = nx.adjacency_matrix(G)
print("undirected graph with self-loops:")
print(A.todense())

# row-stochastic
weights = np.zeros((n, n))

s = A.sum(axis=1)
# print(s)
for i in range(n):
    for j in range(n):
        weights[i, j] = A[i, j] / s[i]
print(weights)

# initialize
max_iter = 100
xx = np.zeros((n, max_iter))
x0 = np.random.rand(n, 1)
x0 = np.multiply(5, x0)
xx[:, 0] = x0[:].T

for i in range(max_iter - 1):
    if i % 10 == 0:
        print("Iteration %d / %d" % (i, max_iter))
    for j in range(n):
        neighbor_j = []
        for k in range(n):
            if A[j, k] == 1:
                neighbor_j.append(k)
        U_j = 0
        for k in neighbor_j:
            U_j = U_j + weights[j, k] * xx[k, i]
        xx[j, i + 1] = U_j

plt.subplot(211)
plt.plot(np.arange(0, max_iter), xx.T)
plt.subplot(212)
nx.draw(G, pos=nx.circular_layout(G), nodecolor='r', edge_color='b')
plt.show()
