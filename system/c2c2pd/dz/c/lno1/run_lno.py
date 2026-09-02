from pyscf import gto, scf

#### C2C2PD complex, L7 dataset ####
path2geo = '../../../C.xyz'
path2chk = '../mf.chk'
path2eri = '../cderi.h5'
path2lno = '../lno_result.out'
path2iao = '../iao.h5'

atoms = ''.join(open(path2geo).readlines()[2:])  # skip the atom count and comment lines
####################################

mol = gto.M(atom=atoms,
            basis='sto6g',
            verbose=4,
            unit='A',
            symmetry=0,
            charge=0,
            spin=0,
            max_memory=20000,
            )

# density fitting: 792 AOs, conventional 4-index ERIs would not fit
mf = scf.RHF(mol).density_fit()
mf.chkfile = path2chk
mf.kernel()

stable = False
while not stable:
    print('mean-field stability test')
    mo_i, _, stable, _ = mf.stability(return_status=True)
    if stable:
        print(f'RHF Energy: {mf.e_tot}, stability {stable}')
        break
    dm = mf.make_rdm1(mo_i, mf.mo_occ)
    mf.kernel(dm0=dm)

from afqmc.lno_afqmc import lno_afqmc, tools

lo_coeff, frag_list, frag_name \
    = tools.iao_fragment(mf, 
                         frag_type='h2heavy', 
                         more_loc='pm',
                         save2= path2iao)

options = {'eql_time': 10,
           'n_prop_steps': 50,
           'n_blocks': 200,
           'n_walkers': 50,
           'max_memory': 10000,
           'mix_precision': True,
           'n_batch': 1,
           'seed': 18,
           'walker_type': 'rhf',
           'trial': 'pt2ccsd',
           }

lno_afqmc.run_afqmc(
    mf,
    lo_coeff,
    frag_list,
    frag_name,
    lno_thresh = 1e-5,
    qmc_options = options,
    chol_cut = 1e-5,
    target_qmc_err = 1e-3,
    run_frag = [0],
    nfrozen = None,
    run_mp = True,
    run_cc = True,
    run_qmc = True,
    plot_las = False,
    )
