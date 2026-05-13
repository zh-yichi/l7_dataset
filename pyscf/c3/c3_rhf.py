import numpy as np
from pyscf import gto, scf, cc

geofile = "../../geometry/c3a/B.xyz"
with open(geofile, 'r') as file:
    atoms = file.read()

mol = gto.M(atom = atoms,
            basis = "ccpvdz",
            verbose=4,
            symmetry=0,
            charge=0,
            spin=0,
            max_memory=80000)

mf = scf.RHF(mol).density_fit()
mf.chkfile = './mf.chk'
mf.init_guess = 'chk'
mf.level_shift = 0.1
mf.max_cycle = 200
mf.kernel()

stable = False
while not stable:
    print(f'mf stability test')
    if not stable:
        mo_i, _, stable,_ = mf.stability(return_status=True)
        dm = mf.make_rdm1(mo_i,mf.mo_occ)
        mf.kernel(dm0=dm)
    elif stable:
        print(f'mf energy: {mf.e_tot}, stability {stable}')
        break
