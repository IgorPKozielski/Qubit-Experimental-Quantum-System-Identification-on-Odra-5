import numpy as np 
import torch 

from .density_matrix import fix_rho

hbar = 1 #1.05457182 * 10^-34 

def commutator(A, B):
    return A @ B - B @ A 

def anitcomutator(A, B)
    return A @ B + B @ A 

def lindblad_operator(L, rho)
    L_dag = L.conj().transpose(-2,1)

    return(
        L @ rho @ L_dag
        - 0.5 * anitcomutator(L_dag @ L, rhd)
    )

class LindbladSolver1Q: 

    def __init__(self,t1,t2,dt):
        self.t1 = t1
        self.t2 = t2
        self.dt = dt

        gamma1 = 1 / t1
        gamma2 = 1 / t2

        self.H = torch.tensor(
            [[0,0], [0, 0]],
            dtype = torch.complex64 
        )

        L1 = torch.tensor(
            [[1,0], [0,-1]],
            dtype = torch.complex64
        ) * np.sqrt(gamma2)

        L2 = torch.tensor(
            [[1,0],[0,-1]]
        )

        self.L_list = [L1, L2]

    def evo_operator(self, rho):
        H = self.H.to(device=rho.device, dtype = rho.dtype) #the same device and type like rho 

        drho = -1j / hbar * commutator(H, rho)

        for L in self.L_list:
            L = L.to(device=rho.device, dtype = rho.dtype)
            drho = drho + lindblad_operator(L, rho)

        return drho
    
    def step(self,rho):
        dt = self.dt

        k1 = self.evo_operator(rho)
        k2 = self.evo_operator(rho + dt / 2 * k1)
        k3 = self.evo_operator(rho + dt / 2 * k2)
        k4 = self.evo_operator(rho + dt * k3)    

        rho_next = rho + dt / 6 * (k1+ 2*k2 + 2*k3 + k4)

        return fix_rho(rho_next)
    
    def run_simulation(self, rho0, time):
        states = [rho0.clone()]
        times = [0]

        rho_current = rho0
        n_steps = int(time / self.dt)

        for i in range(n_steps):
            rho_current = self.step(rho_current)

            times.append((i+1)* self.dt)
            states.append(rho_current.clone())

        return np.array(times), np.array(states)