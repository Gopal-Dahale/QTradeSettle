from typing import List, Tuple, Union, Optional
import itertools as it

import numpy as np
from docplex.mp.advmodel import AdvModel
import networkx as nx
import matplotlib.pyplot as plt
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo

from plot_graph import visualise


# Problem instance 1

num_transactions = 3
num_parties = 3
transcation_weights = np.ones(num_transactions)

G = nx.DiGraph()
G.add_nodes_from(range(num_parties))

currencies = [2., 0., 0.]
securities = [0., 3., 0.]

for k in range(num_parties):
    G.nodes[k]['currency'] = currencies[k]
    G.nodes[k]['security'] = securities[k]

delivery_elist = [
    (1, 0, 2., 0),
    (1, 2, 2., 1),
    (2, 0, 2., 2)
]

payment_elist = [
    (0, 1, 1., 0),
    (2, 1, 1., 1),
    (0, 2, 1., 2)
]

for u, v, w, t in delivery_elist:
    G.add_edge(u, v, delivery=w, transaction=t)

for u, v, w, t  in payment_elist:
    G.add_edge(u, v, payment=w, transaction=t)

G.edges(data=True)
visualise(G, hide=False)

# Problem Formulation

mdl = AdvModel(name="Trade Settlement")
x = np.array([mdl.binary_var(name=f"x_{i}") for i in range(num_transactions)])
objective = mdl.sum(transcation_weights[i] * x[i] for i in range(num_transactions))
mdl.maximize(objective)

# === Helper: net effect on a party for one transaction ===
def transaction_effect(G, party, t):
    dC, dS = 0., 0.
    # payments: currency flows from u->v
    for u,v,data in G.out_edges(data=True):
        if data.get('transaction')==t and 'payment' in data:
            if u==party: dC -= data['payment']
            if v==party: dC += data['payment']
    # deliveries: securities flows from u->v
    for u,v,data in G.in_edges(data=True):
        if data.get('transaction')==t and 'delivery' in data:
            if u==party: dS -= data['delivery']
            if v==party: dS += data['delivery']
    return dC, dS


for p in G.nodes:
    startC = G.nodes[p]['currency']
    startS = G.nodes[p]['security']

    values = np.array([transaction_effect(G,p,t) for t in range(num_transactions)])

    constraints = values.T @ x + np.array([startC, startS])
    print(constraints)
    mdl.add_constraint(constraints[0] >= 0)
    mdl.add_constraint(constraints[1] >= 0)

op = from_docplex_mp(mdl)
print(op.prettyprint())


# Convert to QUBO
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(op)

print(qubo.prettyprint())
