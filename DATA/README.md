# Description of each column in data_all.csv

| Column | Description |
|--------|-------------|
| **mp_id**  | Material Project ID |
| **formula**  | Chemical formula |
| **spg_number**  | Space group number |
| **natoms_prim**  | Number of atoms in primitive cell |
| **natoms_conv**  | Number of atoms in conventional cell |
| **natoms_sc**  | Number of atoms in initial setting supercell |
| **trans_conv2prim**  | Transformation matrix from conventional to primitive cell |
| **trans_conve2sc**  | Transformation matrix from conventional to supercell |
| **structure**  | Structure information (numbers, positions, cell, pbc) |
| **volume**  | Volume of primitive cell [A^3] |
| **nac** | Nonanalytic correction |
| **relax_type**  | Type of relaxation (normal or strict) |
| **kp**  | Peierls lattice thermal conductivity [W/mK] |
| **kc**  | Coherence thermal conductivity [W/mK] |
| **klat**  | Total lattice thermal conductivity (kp + kc) [W/mK] |
| **max_gap**  | Maximum phonon band gap [THz] |
| **gaps**  | Phonon band gaps information between sections [THz] |
| **kcumu_sections**  | Cumulative kappa in each frequency section [W/mK] |
| **max_phfreq**  | Maximum phonon frequency [THz] |
| **phfreq**  | Phonon frequencies for DOS and spectral kappa [THz] |
| **phdos**  | Phonon DOS [arb. unit] |
| **pdos**  | Phonon PDOS for each element [arb. unit] |
| **kspec_freq**  | Spectral kappa w.r.t. frequency [W/mK/THz] |
| **kcumu_freq**  | Cumulative spectral kappa w.r.t. frequency [W/mK] |
| **kspec_norm_freq**  | Normalized spectral kappa w.r.t. frequency [arb. unit] |
| **mfp**  | Phonon mean free path [nm] |
| **log_mfp**  | Log10 of phonon mean free path (in nanometer) |
| **kspec_norm_mfp**  | Normalized spectral kappa w.r.t. MFP [arb. unit] |
| **kcumu_norm_mfp**  | Cumulative normalized spectral kappa w.r.t. MFP [arb. unit] |
| **fc2_error**  | Fitting error for 2nd order force constants |
| **fc3_error**  | Fitting error for 3rd order force constants |

Note
====

- The transformation matrix is defined using the [Phonopy convention](https://phonopy.github.io/phonopy/setting-tags.html#primitive-axes-tag). 
- The anharmonic properties correspond to the values at 300 K.
- Please refer to the [official ALAMODE page](https://alamode.readthedocs.io/en/latest/faq.html) for details on the fitting error and 
[here](https://alamode.readthedocs.io/en/latest/anphondir/inputanphon.html#anphon-nonanalytic) for the nonanalytic correction.

