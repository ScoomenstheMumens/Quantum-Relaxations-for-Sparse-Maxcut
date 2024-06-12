import numpy as np
from qiskit.algorithms import MinimumEigensolver, VQEResult
from qiskit.quantum_info import SparsePauliOp
import cmath
from util import cost_string
import util
import math
import ansatz_circ 
import mthree
from IPython.display import clear_output




class QRAO_encoding_VQE(MinimumEigensolver):
    
    def __init__(self,estimator,sampler, circuit, optimizer,graph,min,shots=None,initial_parameters=None,callback=None):
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._obs=util.operator_vertex_pauli(graph)
        self._num_qubits=self._circuit[0].num_qubits
        self._graph=graph
        self._sampler=sampler
        self._ham=util.edge_pauli(graph)
        
    def compute_minimum_eigenvalue(self,min):
        def objective(x):
            qc=self._circuit[0].bind_parameters(x)
            qc.remove_final_measurements()
            job = self._estimator.run(circuits=qc, observables=self._ham,shots=self._shots)
            H=job.result().values
            if self._callback is not None:

                self._callback([H[0],[x]])
                #print(H)
            return H[0]

        if self.initial_parameters is None:
            x0 = np.pi/2 * np.random.rand(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters

        res = self._optimizer.minimize(objective, x0=x0)

        exp_vertex=[]
        string=[]
        qc=self._circuit[0].bind_parameters(res.x)
        qc.remove_final_measurements()
        for op in self._obs:
            #print(op[0])
            job = self._estimator.run(circuits=qc, observables=op[0],shots=self._shots)
            exp_vertex.append(job.result().values)

        print(res.x)
        print(exp_vertex)
        for exp in exp_vertex:
            if np.sign(exp)<0:
                string+='0'
            else:
                string+='1'

        print(string)
        print(util.cost_string(self._graph,string))
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result



class QRAO_encoding_exp_VQE(MinimumEigensolver):
    
    def __init__(self,estimator,sampler, circuit, optimizer,graph,min,shots=None,initial_parameters=None,callback=None):
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._obs=util.operator_vertex_pauli(graph)
        self._num_qubits=self._circuit[0].num_qubits
        self._graph=graph
        self._sampler=sampler
        self._ham=util.edge_pauli(graph)

        
    def compute_minimum_eigenvalue(self,min):
        def objective(x):
            H=0
            exp_vertex=[]
            string=[]
            qc=self._circuit[0].bind_parameters(x)
            qc.remove_final_measurements()
            for op in self._ham:
                job = self._estimator.run(circuits=qc, observables=op[0],shots=self._shots)
                exp_vertex.append(job.result().values)
                H+=np.real(job.result().values)/2
            if self._callback is not None:

                self._callback([H,[x],exp_vertex])
            return H
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.random.rand(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters
        res = self._optimizer.minimize(objective, x0=x0)

        exp_vertex=[]
        string=[]
        qc=self._circuit[0].bind_parameters(res.x)
        qc.remove_final_measurements()
        for op in self._obs:
            job = self._estimator.run(circuits=qc, observables=op[0],shots=self._shots)
            exp_vertex.append(job.result().values)
        for exp in exp_vertex:
            if np.sign(exp)<0:
                string+='0'
            else:
                string+='1'

        print(string)
        print(util.cost_string(self._graph,string))
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result
    


class QRAO_quadratic_VQE(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,min,alpha,beta,v,gamma,shots=None,initial_parameters=None,callback=None):
        self._graph=graph
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._problem=[]
        self._alpha=alpha
        self._beta=beta 
        self._gamma=gamma
        self._v=v
        for op,coeff in util.operator_vertex(self._graph):
            self._problem.append(op)

    def compute_minimum_eigenvalue(self,min):
        def objective(x):
            circ=[]
            for c in self._circuit:
                a=c.bind_parameters(x)
                circ.append(a)
            jobs= self._estimator.run(circuits=circ)
            N_qubits=len(self._problem[0])
            H=0
            H_round=0
            H_tanh=0
            reg=0

            counts_z= util.counts_in_binary_with_padding(jobs.result().quasi_dists[0],N_qubits)
            counts_x= util.counts_in_binary_with_padding(jobs.result().quasi_dists[1],N_qubits)
            counts_y= util.counts_in_binary_with_padding(jobs.result().quasi_dists[2],N_qubits)
            
            exps_vertex=np.zeros(len(self._graph.nodes()))
            for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
 
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))

            for value in exps_vertex:
                reg+=1/len(self._graph.nodes())*(np.abs(value)-self._gamma)**2  
            for edge in self._graph.edges():
                H-=self._graph[edge[0]][edge[1]].get('weight')*(1-(exps_vertex[int(edge[0])-1])*(exps_vertex[int(edge[1])-1]))/2
                H_tanh-=self._graph[edge[0]][edge[1]].get('weight')*(1-(np.tanh(self._alpha*exps_vertex[int(edge[0])-1]))*np.tanh((self._alpha*exps_vertex[int(edge[1])-1])))/2
                H_round+=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(exps_vertex[int(edge[0])-1])*np.sign(exps_vertex[int(edge[1])-1]))/2
            reg=self._beta*self._v*reg
            if self._callback is not None:
                print('approxs')
                print(-H/self._min)
                print(H_tanh/self._min)
                print(H_round/self._min)
                self._callback([-H,reg,H_round,exps_vertex,H_tanh])
            H+=reg
            return H
    
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.random.rand(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters
        N_qubits=len(self._problem[0])

        res = self._optimizer.minimize(objective, x0=x0)
        job_z= self._estimator.run(circuits=self._circuit[0].bind_parameters(res.x))
        job_x= self._estimator.run(circuits=self._circuit[1].bind_parameters(res.x))
        job_y= self._estimator.run(circuits=self._circuit[2].bind_parameters(res.x))
        H=0

        counts_z= util.counts_in_binary_with_padding(job_z.result().quasi_dists[0],N_qubits)
        counts_x= util.counts_in_binary_with_padding(job_x.result().quasi_dists[0],N_qubits)
        counts_y= util.counts_in_binary_with_padding(job_y.result().quasi_dists[0],N_qubits)
        i=0

        exps_vertex=np.zeros(len(self._graph.nodes()))
        for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))


        for edge in self._graph.edges():
            H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(exps_vertex[int(edge[0])-1])*np.sign(exps_vertex[int(edge[1])-1]))/2
        string=[]
        for i in range(len(self._graph.nodes())):
            if np.sign(self._alpha*exps_vertex[i])<0:
                string+='0'
            else:
                string+='1'
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result
    

class QRAO_nonlinear_VQE(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,min,alpha,beta,v,gamma,shots=None,initial_parameters=None,callback=None):
        self._graph=graph
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._problem=[]
        self._alpha=alpha
        self._beta=beta 
        self._gamma=gamma
        self._v=v
        for op,coeff in util.operator_vertex(self._graph):
            self._problem.append(op)

    def compute_minimum_eigenvalue(self,min):
        def objective(x):
            circ=[]
            for c in self._circuit:
                a=c.bind_parameters(x)
                circ.append(a)
            jobs=self._estimator.run(circuits=circ)
            N_qubits=len(self._problem[0])
            H=0
            H_tanh=0
            H_round=0
            reg=0

            
            counts_z= util.counts_in_binary_with_padding(jobs.result().quasi_dists[0],N_qubits)
            counts_x= util.counts_in_binary_with_padding(jobs.result().quasi_dists[1],N_qubits)
            counts_y= util.counts_in_binary_with_padding(jobs.result().quasi_dists[2],N_qubits)
            
            exps_vertex=np.zeros(len(self._graph.nodes()))
            for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))
            for value in exps_vertex:
                reg+=1/len(self._graph.nodes())*(np.abs(np.tanh(value))-self._gamma)**2  
            for edge in self._graph.edges():
                H-=self._graph[edge[0]][edge[1]].get('weight')*(1-(exps_vertex[int(edge[0])-1])*(exps_vertex[int(edge[1])-1]))/2
                H_tanh-=self._graph[edge[0]][edge[1]].get('weight')*(1-(np.tanh(self._alpha*exps_vertex[int(edge[0])-1]))*np.tanh((self._alpha*exps_vertex[int(edge[1])-1])))/2
                H_round+=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(exps_vertex[int(edge[0])-1])*np.sign(exps_vertex[int(edge[1])-1]))/2
            reg=self._beta*self._v*reg
            if self._callback is not None:
                print('approxs')
                print(-H/self._min)
                print(H_tanh/self._min)
                print(H_round/self._min)
                self._callback([-H,reg,H_round,exps_vertex,H_tanh])
            H_tanh+=reg
            return H_tanh
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.random.rand(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters
        N_qubits=len(self._problem[0])
        res = self._optimizer.minimize(objective, x0=x0)
        job_z= self._estimator.run(circuits=self._circuit[0].bind_parameters(res.x))
        job_x= self._estimator.run(circuits=self._circuit[1].bind_parameters(res.x))
        job_y= self._estimator.run(circuits=self._circuit[2].bind_parameters(res.x))
        H=0
        counts_z= util.counts_in_binary_with_padding(job_z.result().quasi_dists[0],N_qubits)
        counts_x= util.counts_in_binary_with_padding(job_x.result().quasi_dists[0],N_qubits)
        counts_y= util.counts_in_binary_with_padding(job_y.result().quasi_dists[0],N_qubits)
        i=0

        exps_vertex=np.zeros(len(self._graph.nodes()))
        for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))


        for edge in self._graph.edges():
            H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(exps_vertex[int(edge[0])-1])*np.sign(exps_vertex[int(edge[1])-1]))/2
        string=[]
        for i in range(len(self._graph.nodes())):
            if np.sign(self._alpha*exps_vertex[i])<0:
                string+='0'
            else:
                string+='1'
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result
    



class QRAO_initial_state_VQE(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,min,alpha,beta,v,gamma,shots=None,initial_parameters=None,callback=None):
        self._graph=graph
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._problem=[]
        self._alpha=alpha
        self._beta=beta 
        self._gamma=gamma
        self._v=v
        for op,coeff in util.operator_vertex(self._graph):
            self._problem.append(op)

    def compute_minimum_eigenvalue(self,min):
        def objective(x):
            circ=[]
            for c in self._circuit:
                a=c.bind_parameters(x)
                circ.append(a)
            jobs=self._estimator.run(circuits=circ)
            N_qubits=len(self._problem[0])
            H_tanh=0
            reg=0

            
            counts_z= util.counts_in_binary_with_padding(jobs.result().quasi_dists[0],N_qubits)
            counts_x= util.counts_in_binary_with_padding(jobs.result().quasi_dists[1],N_qubits)
            counts_y= util.counts_in_binary_with_padding(jobs.result().quasi_dists[2],N_qubits)
            
            exps_vertex=np.zeros(len(self._graph.nodes()))
            for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))
            for value in exps_vertex:
                reg+=1/len(self._graph.nodes())*(np.abs(np.tanh(value))-self._gamma)**2  
            reg=self._beta*self._v*reg**2
            if self._callback is not None:
                self._callback([reg,exps_vertex])
            H_tanh+=reg
            return H_tanh
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.random.rand(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters
        N_qubits=len(self._problem[0])
        res = self._optimizer.minimize(objective, x0=x0)
        job_z= self._estimator.run(circuits=self._circuit[0].bind_parameters(res.x))
        job_x= self._estimator.run(circuits=self._circuit[1].bind_parameters(res.x))
        job_y= self._estimator.run(circuits=self._circuit[2].bind_parameters(res.x))
        H=0
        counts_z= util.counts_in_binary_with_padding(job_z.result().quasi_dists[0],N_qubits)
        counts_x= util.counts_in_binary_with_padding(job_x.result().quasi_dists[0],N_qubits)
        counts_y= util.counts_in_binary_with_padding(job_y.result().quasi_dists[0],N_qubits)
        i=0

        exps_vertex=np.zeros(len(self._graph.nodes()))
        for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))
        reg=0
        for value in exps_vertex:
                reg+=1/len(self._graph.nodes())*(np.abs(np.tanh(value))-self._gamma)**2  
        reg=self._beta*self._v*reg**2
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = []
        result.eigenvalue = reg
        result.optimal_parameters = res.x
        return result
    


# Define a custome VQE class to orchestra the ansatz, classical optimizers, 
# initial point, callback, and final result
from IPython.display import clear_output
class VQE_full_encoding(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,observables,num_qubits,min,alpha,beta,v,shots=None,initial_parameters=None,callback=None):
        self.graph=graph
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._obs=observables
        self._alpha=alpha
        self._num_qubits=num_qubits
        self._beta=beta 
        self._v=v

        
    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            qc=self._circuit.bind_parameters(x)
            job = self._estimator.run(circuits=[qc]*len(self._obs), parameters=x, observables=self._obs,shots=self._shots)
            H=0
            for edge in self.graph.edges():
                H-=self.graph[edge[0]][edge[1]].get('weight')*(1-np.tanh(self._alpha*job.result().values[edge[0]-1])*np.tanh(self._alpha*job.result().values[edge[1]-1]))/2
            reg=0
            for i in range(0,len(self._obs)):
                reg+=1/self._num_qubits*(np.tanh(self._alpha*job.result().values[i])**2)
            
            reg=self._beta*self._v*reg**2
            H+=reg
            if self._callback is not None:
                self._callback([H,[x]])
            return H
            
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.random.rand(self._circuit.num_parameters)
        else:
            x0=self.initial_parameters
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        qc=self._circuit.bind_parameters(res.x)
        job = self._estimator.run(circuits=[qc]*len(self._obs), parameters=res.x, observables=self._obs,shots=self._shots)
        H=0
        for i,edge in enumerate(self.graph.edges()):
            H-=self.graph[edge[0]][edge[1]].get('weight', None)*(1-np.sign(self._alpha*job.result().values[edge[0]-1])*np.sign(self._alpha*job.result().values[edge[1]-1]))/2
        string=[]
        for i in range(len(self._obs)):
            if np.sign(self._alpha*job.result().values[i])<0:
                string+='0'
            else:
                string+='1'
        print(H)
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self.graph,string)
        result.optimal_parameters = res.x
        return result
    


from IPython.display import clear_output
class VQE_full_encoding_quad(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,observables,num_qubits,min,alpha,beta,v,shots=None,initial_parameters=None,callback=None):
        self.graph=graph
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._obs=observables
        self._alpha=alpha
        self._num_qubits=num_qubits
        self._beta=beta 
        self._v=v

        
    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            qc=self._circuit.bind_parameters(x)
            job = self._estimator.run(circuits=[qc]*len(self._obs), parameters=x, observables=self._obs,shots=self._shots)
            H=0
            for edge in self.graph.edges():
                H-=self.graph[edge[0]][edge[1]].get('weight')*(1-job.result().values[edge[0]-1]*job.result().values[edge[1]-1])/2
            reg=0
            for i in range(0,len(self._obs)):
                reg+=1/self._num_qubits*(np.tanh(self._alpha*job.result().values[i])**2)
            
            reg=self._beta*self._v*reg**2
            H+=reg
            if self._callback is not None:

                self._callback([H,[x]])

            return H
            
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.random.rand(self._circuit.num_parameters)
            
        else:
            x0=self.initial_parameters
        
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        qc=self._circuit.bind_parameters(res.x)
        job = self._estimator.run(circuits=[qc]*len(self._obs), parameters=res.x, observables=self._obs,shots=self._shots)
        H=0
        for i,edge in enumerate(self.graph.edges()):
            H-=self.graph[edge[0]][edge[1]].get('weight', None)*(1-np.sign(self._alpha*job.result().values[edge[0]-1])*np.sign(self._alpha*job.result().values[edge[1]-1]))/2

        string=[]
        for i in range(len(self._obs)):
            if np.sign(self._alpha*job.result().values[i])<0:
                string+='0'
            else:
                string+='1'
        print(H)
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self.graph,string)
        result.optimal_parameters = res.x
        return result