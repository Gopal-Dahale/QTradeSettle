from typing import List, Tuple, Union, Optional
import itertools as it

import numpy as np
from docplex.mp.advmodel import Model
import networkx as nx
import matplotlib.pyplot as plt
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from plot_graph import visualise
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

SEED = 12345

# Problem instance 1

# num_transactions = 3
# num_parties = 3
# transcation_weights = np.ones(num_transactions)

# G = nx.DiGraph()
# G.add_nodes_from(range(num_parties))

# currencies = [2., 0., 0.]
# securities = [0., 3., 0.]

# for k in range(num_parties):
#     G.nodes[k]['currency'] = currencies[k]
#     G.nodes[k]['security'] = securities[k]

# delivery_elist = [
#     (1, 0, 2., 0),
#     (1, 2, 2., 1),
#     (2, 0, 2., 2)
# ]

# payment_elist = [
#     (0, 1, 1., 0),
#     (2, 1, 1., 1),
#     (0, 2, 1., 2)
# ]

# for u, v, w, t in delivery_elist:
#     G.add_edge(u, v, delivery=w, transaction=t)

# for u, v, w, t  in payment_elist:
#     G.add_edge(u, v, payment=w, transaction=t)

# G.edges(data=True)
# visualise(G, hide=False)

# Problem instance 2
num_transactions = 7
num_parties = 6
transcation_weights = np.ones(num_transactions)

G = nx.DiGraph()
G.add_nodes_from(range(num_parties))

currencies = np.array([5., 1., 2., 3, 2, 1])
securities = np.zeros(num_parties)

for k in range(num_parties):
    G.nodes[k]['currency'] = currencies[k]
    G.nodes[k]['security'] = securities[k]

delivery_elist = []

payment_elist = [
    (0, 1, 4., 0),
    (0, 2, 3., 1),
    (0, 3, 2., 2),
    (1, 4, 3., 3),
    (1, 5, 3., 4),
    (4, 5, 6., 5),
    (5, 4, 4., 6)
]

for u, v, w, t in delivery_elist:
    G.add_edge(u, v, delivery=w, transaction=t)

for u, v, w, t  in payment_elist:
    G.add_edge(u, v, payment=w, transaction=t)

G.edges(data=True)
g = visualise(G, hide=False)

type(g)

# Problem Formulation

mdl = Model(name="Trade Settlement")
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


constants = []
linear_parts = []

mdl_qubo = Model(name="Trade Settlement QUBO")
x_qubo = np.array([mdl_qubo.binary_var(name=f"x_{i}") for i in range(num_transactions)])

for p in G.nodes:
    startC = G.nodes[p]['currency']
    startS = G.nodes[p]['security']

    values = np.array([transaction_effect(G,p,t) for t in range(num_transactions)])

    constraints = values.T @ x + np.array([startC, startS])
    
    mdl.add_constraint(constraints[0] >= 0)
    mdl.add_constraint(constraints[1] >= 0)
    
    linear_parts += [values.T[0], values.T[1]]
    constants += [-startC, -startS]


# solve exactly using CPLEX
mdl.parameters.randomseed = SEED
result = mdl.solve()

print("Objective value:", result.objective_value)
print("Values:", np.array(result.get_values(x), dtype=int))

op = from_docplex_mp(mdl)
print(op.prettyprint())


# QAOA with subgradient descent


################### Backend and transpiler setup ###################
backend_options = {
    "precision": "single",
    "max_parallel_threads": 12,
    "max_parallel_experiments": 0,
    "max_parallel_shots": 0,
    "fusion_enable": True,
    "fusion_threshold": 14,
    "fusion_max_qubit": 4,
    "matrix_product_state_max_bond_dimension": 16,
    "matrix_product_state_truncation_threshold": 1e-8,
    "mps_sample_measure_algorithm": "mps_apply_measure",
    "mps_parallel_threshold": 12,
    "mps_omp_threads": 12,
    "mps_lapack": False,
    "seed_simulator": SEED,
}

aer_sim = AerSimulator(method="matrix_product_state", **backend_options)
aer_sim.set_max_qubits(200)

# Create pass manager for transpilation
pm = generate_preset_pass_manager(
    backend=aer_sim, optimization_level=1, seed_transpiler=SEED
)

def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
 
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
 
    results = job.result()[0]
    cost = results.data.evs
 
    objective_func_vals.append(cost)
 
    return cost

# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]
 

estimator = Estimator(mode=aer_sim)
estimator._backend = aer_sim
estimator.options.default_shots = 10000

 # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `backend=`
sampler = Sampler(mode=aer_sim)
sampler._backend = aer_sim
sampler.options.default_shots = 10000

######################################################################

np.random.seed(SEED)
lam = np.random.uniform(10, 20, size=len(constants)) # penalty coefficients

initial_gamma = np.pi
initial_beta = np.pi / 2
init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]

print(op.prettyprint())

best_solution = None
best_obj_val = -np.inf

for step in range(20):

    print("Step:", step)
    print("Lagrange multipliers:", lam)

    # quadratic program to QUBO with Lagrangian dualization
    # minimizing hence minus sign
    qubo_obj = -mdl.sum(transcation_weights[i] * x_qubo[i] for i in range(num_transactions))

    for i, (c, l) in enumerate(zip(constants, linear_parts)):
        qubo_obj += lam[i] * (c - l @ x_qubo)

    mdl_qubo.minimize(qubo_obj)

    # Convert to QUBO
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(from_docplex_mp(mdl_qubo))

    # print(qubo.prettyprint())

    qubit_op, offset = qubo.to_ising()

    circuit = QAOAAnsatz(cost_operator=qubit_op, reps=2)
    circuit.measure_all()
 
    qc = pm.run(circuit)
    qc.draw("mpl", fold=False, idle_wires=False)

    objective_func_vals = []  # Global variable

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(qc, qubit_op, estimator),
        method="COBYLA",
        tol=1e-3,
    )

    init_params = result.x + np.random.uniform(-0.1, 0.1, size=len(init_params))

    optimized_circuit = qc.assign_parameters(result.x)
    
    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=int(1e4))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val / shots for key, val in counts_int.items()}
    print()

    keys = list(final_distribution_int.keys())
    values = list(final_distribution_int.values())
    most_likely = keys[np.argmax(np.abs(values))]
    most_likely_bitstring = to_bitstring(most_likely,qubit_op.num_qubits)
    most_likely_bitstring.reverse()
    
    print("Result bitstring:", most_likely_bitstring)

    sub = op.substitute_variables(constants=dict(zip([f"x_{i}" for i in range(num_transactions)], most_likely_bitstring)))

    print(sub.status)

    if sub.status.value == 0: # feasible
        obj_eval = op.objective.evaluate(most_likely_bitstring)

        if obj_eval > best_obj_val:
            best_obj_val = obj_eval
            best_solution = most_likely_bitstring
        

    # Update Lagrange multipliers

    alpha = 0.1
    for i, (c, l) in enumerate(zip(constants, linear_parts)):
        s = c - l @ np.array(most_likely_bitstring)
        lam[i] = max(0, lam[i] + alpha * s)  # subgradient descent update

print("Best objective value:", best_obj_val)
print("Best solution:", best_solution)