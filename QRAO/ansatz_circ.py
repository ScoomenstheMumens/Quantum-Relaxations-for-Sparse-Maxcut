
from qiskit import  QuantumCircuit, QuantumRegister,ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate as XY
import numpy as np
import util
def sep_ansatz_scal(graph,depth):
    theta = ParameterVector("m",1000)
    encoded_problem=util.graph_to_paulis(graph)
    N_qubits=encoded_problem[-1][2][-1]+1
    qr=QuantumRegister(N_qubits)
    qc=QuantumCircuit(qr)
    print(N_qubits)
    index=0
    for color in encoded_problem:
        qr_aux=[]
        for qubit in color[2]:
            qr_aux.append(qr[qubit])

        num_qubits=len(color[2])
        for j in range (0,depth): 

            for k in range (num_qubits):
                qc.ry(theta[index],qr_aux[k])
                index+=1
            for k in range (num_qubits):
                qc.rz(theta[index],qr_aux[k])
                index+=1
                '''
            for k in range (int((num_qubits)/2)):
                qc.append(XY(theta[index],np.pi),[qr_aux[2*k],qr_aux[2*k+1]])
                index+=1
            for k in range (int((num_qubits)/2)):
                if 2*k+1<num_qubits and 2*k+2<num_qubits:
                    qc.append(XY(theta[index],np.pi),[qr_aux[2*k+1],qr_aux[2*k+2]])
                    index+=1
                    '''
            for k in range(num_qubits-1):
                qc.cx(qr_aux[k],qr_aux[k+1])

    return qc



def sep_ansatz(graph,depth):
    theta = ParameterVector("m",1000)
    encoded_problem=util.graph_to_paulis_qrao(graph)
    N_qubits=encoded_problem[-1][2][-1]+1
    qr=QuantumRegister(N_qubits)
    qc=QuantumCircuit(qr)
    print(N_qubits)
    index=0
    for color in encoded_problem:
        qr_aux=[]
        for qubit in color[2]:
            qr_aux.append(qr[qubit])

        num_qubits=len(color[2])
        for j in range (0,depth): 

            for k in range (num_qubits):
                qc.ry(theta[index],qr_aux[k])
                index+=1
            for k in range (num_qubits):
                qc.rz(theta[index],qr_aux[k])
                index+=1
                '''
            for k in range (int((num_qubits)/2)):
                qc.append(XY(theta[index],np.pi),[qr_aux[2*k],qr_aux[2*k+1]])
                index+=1
            for k in range (int((num_qubits)/2)):
                if 2*k+1<num_qubits and 2*k+2<num_qubits:
                    qc.append(XY(theta[index],np.pi),[qr_aux[2*k+1],qr_aux[2*k+2]])
                    index+=1
                    '''
            for k in range(num_qubits-1):
                qc.cx(qr_aux[k],qr_aux[k+1])

    return qc

def ansatz_xy(num_qubits,depth):
  theta = ParameterVector("m",3*num_qubits*depth-depth)
  qr=QuantumRegister(num_qubits)
  circuit = QuantumCircuit(qr)

  index=0
  for j in range (0,depth): 
    for i in range (num_qubits):
      circuit.ry(theta[index],qr[i])
      index+=1
    for i in range (num_qubits):
      circuit.rz(theta[index],qr[i])
      index+=1
    for i in range (int((num_qubits)/2)):
      circuit.append(XY(theta[index],np.pi),[qr[2*i],qr[2*i+1]])
      index+=1
    for i in range (int((num_qubits)/2)):
      if 2*i+1<num_qubits and 2*i+2<num_qubits:
        circuit.append(XY(theta[index],np.pi),[qr[2*i+1],qr[2*i+2]])
        index+=1
    circuit.barrier()
  return circuit

def ansatz_efficient(num_qubits,depth):
  theta = ParameterVector("m",3*num_qubits*depth-depth)
  qr=QuantumRegister(num_qubits)
  circuit = QuantumCircuit(qr)

  index=0
  for j in range (0,depth): 
    for i in range (num_qubits):
      circuit.rx(theta[index],qr[i])
      index+=1
    for i in range (num_qubits):
      circuit.rz(theta[index],qr[i])
      index+=1
    for i in range(num_qubits-1):
        circuit.cx(qr[i],qr[i+1])
    circuit.barrier()
  return circuit



def multibasis_ansatz(ansatz):  
    num_qubits=ansatz.num_qubits
    qr_z=QuantumRegister(num_qubits)
    cr_z=ClassicalRegister(num_qubits)
    circuit_z = QuantumCircuit(qr_z,cr_z)
    circuit_z.append(ansatz,qr_z)


    qr_x=QuantumRegister(num_qubits)
    cr_x=ClassicalRegister(num_qubits)
    circuit_x = QuantumCircuit(qr_x,cr_x)
    circuit_x.append(ansatz,qr_x)
    circuit_x.h(qr_x[i] for i in range (ansatz.num_qubits))
    

    qr_y=QuantumRegister(num_qubits)
    cr_y=ClassicalRegister(num_qubits)
    circuit_y = QuantumCircuit(qr_y,cr_y)
    circuit_y.append(ansatz,qr_y)
    circuit_y.h(qr_y[i] for i in range (ansatz.num_qubits))
    circuit_y.s(qr_y[i] for i in range (ansatz.num_qubits))
    
    for i in range (num_qubits):
      circuit_z.measure(qr_z[i],cr_z[i])
      circuit_x.measure(qr_x[i],cr_x[i])
      circuit_y.measure(qr_y[i],cr_y[i])



    return circuit_z,circuit_x,circuit_y
def multibasis_ansatz_maximally(ansatz):  
    num_qubits=ansatz.num_qubits
    
    qr_z=QuantumRegister(2*num_qubits)
    qr_aux_z=[]
    for i in range (num_qubits):
      qr_aux_z.append(qr_z[i])
    
    cr_z=ClassicalRegister(num_qubits)
    circuit_z = QuantumCircuit(qr_z,cr_z)
    for i in range (num_qubits):
        circuit_z.ry(np.pi/2+0.2,qr_z[2*num_qubits-1-i])
        circuit_z.cx(qr_z[2*num_qubits-1-i],qr_z[i])
    circuit_z.append(ansatz,qr_aux_z)


    qr_x=QuantumRegister(2*num_qubits)
    qr_aux_x=[]
    for i in range (num_qubits):
      qr_aux_x.append(qr_x[i])
    
    cr_x=ClassicalRegister(num_qubits)
    circuit_x = QuantumCircuit(qr_x,cr_x)
    
    for i in range (num_qubits):
        circuit_x.ry(np.pi/2+0.1,qr_x[2*num_qubits-1-i])
        circuit_x.cx(qr_x[2*num_qubits-1-i],qr_x[i])
    circuit_x.append(ansatz,qr_aux_x)
    circuit_x.h(qr_x[i] for i in range (ansatz.num_qubits))
    

    qr_y=QuantumRegister(2*num_qubits)
    qr_aux_y=[]
    for i in range (num_qubits):
      qr_aux_y.append(qr_y[i])
    
    cr_y=ClassicalRegister(num_qubits)
    circuit_y = QuantumCircuit(qr_y,cr_y)
    for i in range (num_qubits):
        circuit_y.ry(np.pi/2+0.05,qr_y[2*num_qubits-1-i])
        circuit_y.cx(qr_y[2*num_qubits-1-i],qr_y[i])
    circuit_y.append(ansatz,qr_aux_y)
    circuit_y.h(qr_y[i] for i in range (ansatz.num_qubits))
    circuit_y.s(qr_y[i] for i in range (ansatz.num_qubits))
    
    for i in range (num_qubits):
      circuit_z.measure(qr_z[i],cr_z[i])
      circuit_x.measure(qr_x[i],cr_x[i])
      circuit_y.measure(qr_y[i],cr_y[i])



    return circuit_z,circuit_x,circuit_y


