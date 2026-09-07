import numpy as np
from pyscf import gto, scf, lo

lno_num = 2
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

from afqmc.lno_afqmc import lno_afqmc, tools
from pyscf.data import elements
lo_coeff, frag_lolist, atm_center = tools.iao_localization(mf)

options = {
           'n_eql': 50,
           'n_blocks': 600,
           'n_walkers': 300,
           'max_memory': 20000,
           'mix_precision': True,
           'n_batch': 1,
           'seed': 17,
           'walker_type': 'rhf',
           'trial': 'pt2ccsd',
           }

lno_afqmc.run_afqmc(
              mf,
              lo_coeff = lo_coeff,
              frag_lolist = frag_lolist,
              nfrozen = elements.chemcore(mol),
              thresh = lno_thresh,
              qmc_options = options,
              chol_cut = 1e-5,
              target_sto_error = 5e-4,
              atom_group = atm_center,
              run_frg_list = None,
              )
