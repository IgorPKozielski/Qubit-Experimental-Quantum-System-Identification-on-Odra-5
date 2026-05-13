import numpy as np 
import torch 

from .density_matrix import fix_rho

hbar = 1 #1.05457182 * 10^-34 
#handy 
def commutator(A, B):
    return A @ B - B @ A 

def anticommutator(A, B):
    return A @ B + B @ A 
#Lindblad dissipator (single)
def lindblad_operator(L, rho):
    L_dag = L.conj().transpose(-2,-1)   #hermitian conjugate of the L* 
#D[L](rho) = L rho L* -1/2 {L*L.rho}
    return(
        L @ rho @ L_dag
        - 0.5 * anticommutator(L_dag @ L, rho)
    )
#1-qubit lindblad Solver
class LindbladSolver1Q: 

    def __init__(self,t1,t2,dt):
        #store relaxation/dephasing times and simulation step
        self.t1 = t1
        self.t2 = t2
        self.dt = dt
#times to rates
        gamma1 = 1 / t1
        gamma2 = 1 / t2
#hamiltonian for this simple model (baseline)
        self.H = torch.tensor(
            [[0,0], [0, 0]],            #H=0 so dynamics is due to dissipation
            dtype = torch.complex64 
        )
#|1> -> |0>, T1 like relaxation, amplitude dumping
        L1 = torch.tensor(
            [[0, 1],
            [0, 0]],
            dtype=torch.complex64
        ) * np.sqrt(gamma1)

#destroy coherence without changing populations, dephasing operator 
        L2 = torch.tensor(
            [[1, 0],
            [0, -1]],
            dtype=torch.complex64
        ) * np.sqrt(gamma2)
        self.L_list = [L1, L2]      #list of all lindblad operators 

    def evo_operator(self, rho):
        H = self.H.to(device=rho.device, dtype = rho.dtype) #the same device and type like rho 

        drho = -1j / hbar * commutator(H, rho) 
#dissipative part: sum_k D[L,k](rho)
        for L in self.L_list:
            L = L.to(device=rho.device, dtype = rho.dtype)

            drho = drho + lindblad_operator(L, rho) 

        return drho
    
    def step(self,rho):
        dt = self.dt    #one runge kutta 4 step for d rho /dt = L(rho)

        k1 = self.evo_operator(rho)
        k2 = self.evo_operator(rho + dt / 2 * k1)
        k3 = self.evo_operator(rho + dt / 2 * k2)
        k4 = self.evo_operator(rho + dt * k3)    

        rho_next = rho + dt / 6 * (k1+ 2*k2 + 2*k3 + k4)

        return fix_rho(rho_next)        #enforce Hermiticity, positivity and Tr(rho)=1 
    
    def run_simulation(self, rho0, time):
        states = [rho0.clone()] #store initial state
        times = [0]             #and time

        rho_current = rho0
        n_steps = int(time / self.dt)
    #propagate rho step by step
        for i in range(n_steps):
            rho_current = self.step(rho_current)

            times.append((i+1)* self.dt)
            states.append(rho_current.clone())

        return np.array(times), np.array(states)