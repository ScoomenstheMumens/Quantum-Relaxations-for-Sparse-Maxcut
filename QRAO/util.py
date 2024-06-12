import networkx as nx
import random
import numpy as np 
import random
from qiskit.quantum_info import SparsePauliOp
from itertools import combinations, product
import math
import cvxgraphalgs as cvxgr
from cvxgraphalgs.structures import Cut
from cylp.cy import CyCbcModel, CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray




def cost_string(G, string):
    C = 0
    for edge in G.edges():
        i = int(edge[0])
        j = int(edge[1])
        w = G[edge[0]][edge[1]]["weight"]
        C += w*(1-(int(string[i-1])-1/2)*(int(string[j-1])-1/2)*4)/2
    return C

def Hamiltonian_spectrum(G):
    n=len(G.nodes)
    bit_strings = [bin(i)[2:].zfill(n) for i in range(2**n)]
    result = {bit_string: cost_string(G,bit_string) for bit_string in bit_strings}
    return result


def counts_in_binary_with_padding(counts, n):
    # Step 1: Convert to binary
    bin_counts={}
    for num,count in counts.items():
        binary_representation = bin(num)[2:]  # [2:] removes the '0b' prefix
        
        # Step 2: Add leading zeros if necessary
        if len(binary_representation) < n:
            binary_representation = '0' * (n - len(binary_representation)) + binary_representation
        bin_counts[binary_representation]=count
    
    return bin_counts

def gw_cut(graph):
    sdp_cut = cvxgr.algorithms.goemans_williamson_weighted(graph)
    gw_cut=sdp_cut.evaluate_cut_size(graph)
    gw_string=sdp_cut.left
    return gw_cut




def branch_and_bound(G, num_threads=4):
    N = len(G)
    model = CyLPModel()
    # Decision variables, one for each node
    x = model.addVariable('x', N, isInt=True)
    # Adjacency matrix (possibly weighted)
    W = nx.adjacency_matrix(G).todense()

    z_ind = dict()
    ind = 0
    w = []
    for i in range(N):
        j_range = range(N)
        if (not nx.is_directed(G)):
            # Reduced range for undirected graphs
            j_range = range(i, N)
        for j in j_range:
            if (W[i,j] == 0):
                continue
            if (i not in z_ind):
                z_ind[i] = dict()  
            z_ind[i][j] = ind
            w.append(W[i,j])
            ind += 1
    # Aux variables, one for each edge
    z = model.addVariable('z', len(w), isInt=True)
    # Adding the box contraints
    model += 0 <= x <= 1
    model += 0 <= z <= 1
    # Adding the cutting constraints
    # If x_i == x_j then z_ij = 0
    # If x_i != x_j then z_ij = 1
    for i in z_ind:
        for j in z_ind[i]:
            model += z[z_ind[i][j]] - x[i] - x[j] <= 0
            model += z[z_ind[i][j]] + x[i] + x[j] <= 2
    # Adding the objective function
    model.objective = CyLPArray(w) * z
    lp = CyClpSimplex(model)
    lp.logLevel = 0
    lp.optimizationDirection = 'max'
    mip = lp.getCbcModel()
    mip.logLevel = 0
    # Setting number of threads
    mip.numberThreads = num_threads
    mip.solve()

    return mip.objectiveValue, [int(i) for i in mip.primalVariableSolution['x']]















def generate_pauli_strings(n):
    pauli_set = ['I', 'X', 'Y', 'Z']
    pauli_strings = []
    
    # Generate strings with k non-identity Pauli operators
    for k in range(1, n + 1):
        for positions in combinations(range(n), k):
            for pauli_operators in product(pauli_set[1:], repeat=k):  # Exclude 'I'
                string = ['I'] * n
                for pos, op in zip(positions, pauli_operators):
                    string[pos] = op
                pauli_strings.append(''.join(string))
    
    return pauli_strings


def nodes_by_color(color_dict):
    nodes_by_color_dict = {}
    for node, color in color_dict.items():
        if color not in nodes_by_color_dict:
            nodes_by_color_dict[color] = []
        nodes_by_color_dict[color].append(node)
    return nodes_by_color_dict


def vertex_to_pauli_qrao(num_vertices,num_qubits,k):
    strings=[]
    for i in range (1,k+1):
        if len(strings)<num_vertices:
            strings+=generate_pauli_strings(num_qubits)
    if num_vertices>len(strings):
        raise ValueError("Invalid value for k")
    
    return strings[:num_vertices]

def graph_to_paulis_qrao(graph):
    graph_coloring = nx.greedy_color(graph)
    dict=nodes_by_color(graph_coloring)
    list_aux=[]
    qubits=0
    for i,element in enumerate(dict.values()):

        if len(element)==1:
            list_aux.append([i,element,[qubits],'Z'])
            qubits+=1
        else:
            n_qubits=math.floor(math.log(len(element),4))+1

            list_aux.append([i,element,list(np.arange(qubits,qubits+n_qubits)),vertex_to_pauli_qrao(len(element),n_qubits,n_qubits)])
            qubits+=n_qubits
    return list_aux

def operator_vertex_qrao(graph):
    list_problem=graph_to_paulis_qrao(graph)
    ops_vertex=[None]*len(graph.nodes())
    N_qubits=list_problem[-1][2][-1]+1
    for color in list_problem:
        for i in range (len(color[1])):
            string=color[-1][i]
            new_string=list('I' * (N_qubits))
            string=list(string)

            for j,qubit in enumerate(color[2]):
                if len(color[2])==1:
                    new_string[color[2][0]]=string[0]
                else:
                    new_string[qubit]=string[j]
            new_string=''.join(new_string)
            ops_vertex[int(color[1][i])-1]=[new_string,len(color[1])]
    return ops_vertex

def operator_vertex_pauli(graph):
    obs=[]
    for o,coeff in operator_vertex_qrao(graph):
        obs.append(SparsePauliOp(o,coeffs=np.sqrt(coeff)))
        #obs.append(SparsePauliOp(o,coeffs=1))
    #print(obs)
    return obs


def edge_pauli(graph):
    obs=operator_vertex_pauli(graph)
    edge_obs=0
    for edge in graph.edges():
        edge_obs+=SparsePauliOp(obs[int(edge[0])-1]@obs[int(edge[1])-1],coeffs=graph[edge[0]][edge[1]].get('weight')*obs[int(edge[0])-1].coeffs*obs[int(edge[1])-1].coeffs)
    #print(edge_pauli)
    return edge_obs


















def generate_binary_strings(n, k, x):
    if k < 0 or k > n:
        raise ValueError("Invalid value for k")

    # Generate all combinations of indices with exactly k ones
    indices_combinations = list(combinations(range(n), k))

    # Generate binary strings based on the selected indices
    z_strings = []
    x_strings = []
    y_strings = []
    for indices in indices_combinations:
        binary_string_x= ['I'] * n
        binary_string_y= ['I'] * n
        binary_string_z= ['I'] * n
        for index in indices:
            binary_string_z[index] = 'Z'
            binary_string_y[index] = 'Y'
            binary_string_x[index] = 'X'
        '''
        z_strings.append("".join(binary_string_z))
        #y_strings.append("".join(binary_string_y))
        x_strings.append("".join(binary_string_x))
        '''
        z_strings.append("".join(['I'] * x) + "".join(binary_string_z))
        x_strings.append("".join(['I'] * x) + "".join(binary_string_x))
        y_strings.append("".join(['I'] * x) + "".join(binary_string_y))

    return z_strings + x_strings + y_strings

def vertex_to_pauli_dict(num_vertices,num_qubits,k):
    strings=[]
    for i in range (1,k+1):
        if len(strings)<num_vertices:
            strings+=generate_binary_strings(num_qubits, i,0)
    if num_vertices>len(strings):
        raise ValueError("Invalid value for k")
    hamiltonian_dict={}
    
    for i in range(num_vertices):
        hamiltonian_dict[strings[i]]=1
    print(hamiltonian_dict)
    
    return hamiltonian_dict

def vertex_to_pauli(num_vertices,num_qubits,k):
    strings=[]
    for i in range (1,k+1):
        strings+=generate_binary_strings(num_qubits, i,0)
    if num_vertices>len(strings):
        raise ValueError("Invalid value for k")
    '''
    hamiltonian_dict={}
    
    for i in range(num_vertices):
        hamiltonian_dict[strings[i]]=1
    '''
    
    return strings[:num_vertices]

def graph_to_paulis(graph):
    graph_coloring = nx.greedy_color(graph)
    dict=nodes_by_color(graph_coloring)
    list_aux=[]
    qubits=0
    for i,element in enumerate(dict.values()):

        if len(element)==1:
            list_aux.append([i,element,[qubits],'Z'])
            qubits+=1
        else:

            n_qubits=math.ceil((math.log((len(element)/3+1),2)))

            list_aux.append([i,element,list(np.arange(qubits,qubits+n_qubits)),vertex_to_pauli(len(element),n_qubits,n_qubits)])
            qubits+=n_qubits
    return list_aux

def operator_vertex(graph):
    list_problem=graph_to_paulis(graph)
    ops_vertex=[None]*len(graph.nodes())
    N_qubits=list_problem[-1][2][-1]+1
    for color in list_problem:
        for i in range (len(color[1])):
            string=color[-1][i]
            new_string=list('I' * (N_qubits))
            string=list(string)

            for j,qubit in enumerate(color[2]):
                if len(color[2])==1:
                    new_string[color[2][0]]=string[0]
                else:
                    new_string[qubit]=string[j]
            new_string=''.join(new_string)
            ops_vertex[int(color[1][i])-1]=[new_string,len(color[1])]
    return ops_vertex


def vertex_to_pauli_full_encoding(num_vertices,num_qubits,k,num_ancillas):
    strings=[]
    for i in range (1,k+1):
        strings+=generate_binary_strings(num_qubits, i,num_ancillas)
    print(len(strings))
    if num_vertices>len(strings):
        raise ValueError("Invalid value for k")
    hamiltonian_dict={}
    
    for i in range(num_vertices):
        hamiltonian_dict[strings[i]]=1
    print(hamiltonian_dict)
    '''
    random_elements = random.sample(strings, num_vertices)
    print(random_elements)
    for i in range(num_vertices):
        hamiltonian_dict[random_elements[i]]=1
    '''
    obs=[]
    for o in hamiltonian_dict.items():
        obs.append(SparsePauliOp([o[0]],coeffs=o[1]))
    
    return obs