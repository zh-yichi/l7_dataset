from pyscf import gto, scf, mp, cc

#### C2C2PD complex, L7 dataset ####
path2geo = '../../../C.xyz'
path2chk = '../mf.chk'
path2eri = '../cderi.h5'

atoms = ''.join(open(path2geo).readlines()[2:])  # skip the atom count and comment lines
####################################

mol = gto.M(atom=atoms,
            basis='ccpvdz',
            verbose=4,
            unit='A',
            symmetry=0,
            charge=0,
            spin=0,
            max_memory=400000,
            )

# density fitting: 792 AOs, conventional 4-index ERIs would not fit
mf = scf.RHF(mol).density_fit()
# mf.with_df._cderi_to_save = path2eri
mf.with_df._cderi = path2eri
mf.chkfile = path2chk
mf.init_guess = 'chk'
mf.kernel()

#stable = False
#while not stable:
#    print('mean-field stability test')
#    mo_i, _, stable, _ = mf.stability(return_status=True)
#    if stable:
#        print(f'RHF Energy: {mf.e_tot}, stability {stable}')
#        break
#    dm = mf.make_rdm1(mo_i, mf.mo_occ)
#    mf.kernel(dm0=dm)

# mymp = mp.MP2(mf).set_frozen()
# mymp.kernel()

mycc = cc.CCSD(mf).set_frozen()
mycc.incore_complete = True
mycc.diis_space = 2
mycc.conv_tol = 1e-6
mycc.conv_tol_normt = 3e-5
mycc.kernel()

from afqmc import integral
integral.prep_integral(mycc)
