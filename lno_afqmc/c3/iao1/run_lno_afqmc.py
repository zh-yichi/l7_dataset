import numpy as np
from pyscf import gto, scf, lo

lno_num = 1
lno_list = [3e-5,1e-5,3e-6,1e-6]
lno_thresh = lno_list[lno_num-1]

geofile = "../../../geometry/c3a/B.xyz"
with open(geofile, 'r') as file:
    atoms = file.read()

mol = gto.M(atom=atoms,
            basis="ccpvdz",
            verbose=4,
            unit='angstrom',
            symmetry=0,
            charge=0,
            spin=0,
            max_memory=40000,
            )

mf = scf.RHF(mol).density_fit()
mf.chkfile = '../../../pyscf/c3/mf.chk'
mf.init_guess = 'chk'
mf.kernel()

from pyscf.lno.tools import autofrag_iao
from pyscf import lo
import numpy as np
from pyscf.data import elements

frozen = elements.chemcore(mol)

# IAO localization
orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
lo_coeff = lo.iao.iao(mol, orbocc)
lo_coeff = lo.orth.vec_lowdin(lo_coeff, mf.get_ovlp())
moliao = lo.iao.reference_mol(mol)
frag_lolist = autofrag_iao(moliao)

np.savez('../lo_coeff.npz', lo_coeff_a=lo_coeff)

# load local orbitals
#lo_coeff_a = np.load('../lo_coeff.npz')["lo_coeff_a"]
#lo_coeff_b = np.load('../lo_coeff.npz')["lo_coeff_b"]
#lo_coeff_a = np.array(lo_coeff_a)
#lo_coeff_b = np.array(lo_coeff_b)
#lo_coeff = [lo_coeff_a, lo_coeff_b]
#print('alpha local orbitals shape: ', lo_coeff_a.shape)
#print(' beta local orbitals shape: ', lo_coeff_b.shape)

from afqmc.lno_afqmc import lno_afqmc
options = {
           'n_eql': 100,
           'n_prop_steps': 50,
           'n_blocks': 500,
           'n_walkers': 300,
           'nchol_chunk': 100,
           'mix_precision': True,
           'n_batch': 1,
           'seed': 17,
           'walker_type': 'rhf',
           'trial': 'ccsd_pt2',
           'dt':0.005,
           'use_gpu': True,
           }

lno_afqmc.run_afqmc(
              mf,
              lo_coeff = lo_coeff,
              frag_lolist = frag_lolist,
              nfrozen = frozen,
              thresh = lno_thresh,
              qmc_options = options,
              chol_cut = 1e-5,
              target_sto_error = 5e-4,
              atom_group = moliao.elements,
              )
