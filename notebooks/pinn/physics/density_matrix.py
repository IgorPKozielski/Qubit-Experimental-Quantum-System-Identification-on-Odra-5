import torch

#we need to fix density matrix to ensure physicality
def fix_rho(rho):
    #handle single matrix vs batched input
    squeeze = False

    if rho.ndim == 2:
        squeeze = True
        rho = rho.unsqueeze(0)
    #we need to ensure hermiticity
    rho_dag = rho.conj().transpose(1, 2)
    rho = (rho + rho_dag) / 2
    #rho = V D V*
    eigvals, eigvecs = torch.linalg.eigh(rho)
    #removing negative eigenvalues
    eigvals = torch.clamp(eigvals, min=0)
    #reconstruction of density matrix from eigvals and eigvecs
    D = torch.diag_embed(eigvals.to(eigvecs.dtype))


    rho_fixed = eigvecs @ D @ eigvecs.conj().transpose(1, 2)
    #normalize trace
    trace = rho_fixed.diagonal(dim1=1, dim2=2).sum(dim=1)

    rho_fixed = rho_fixed / trace[:, None, None]
    #return original shape if input was a sigle density matrix
    if squeeze:
        return rho_fixed.squeeze(0)

    return rho_fixed

#compute matrix difference between two density matrices 
def rho_diff(rho1, rho2):

    diff = rho1 - rho2
    #sum of absolute value over matrix dimensions
    return torch.sum(torch.abs(diff), dim=(1, 2))