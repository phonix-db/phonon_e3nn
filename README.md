# Data and code repository for anharmonic phonon property data

This repository provides anharmonic phonon property dataset obtained with first-principles calculations 
and Python scripts for machine learning predictions. 
These data and code were used in arXiv:2504.21245 (2025).

## How to Obtain the Data and Code

To obtain the data and code, download as a zip file or use ``git clone`` command:

```sh
git clone https://github.com/masato1122/phonon_e3nn.git
cd phonon_e3nn
```

## Anharmonic phonon dataset

The anharmonic phonon dataset can be found in `./phonon_e3nn/DATA/data_all.csv`, 
which contains phonon data for approximately 7,000 materials.
This dataset is used for anharmonic phonon analysis.

## How to Use

To run a simple example, navigate to the example directory and execute the shell script.

``` sh
cd ./phonon_e3nn/example
sh run_example.sh
```

## Citation

If you use the phonon data from this repository, which is stored in `./phonon_e3nn/DATA`, please cite the following paper:

- **[Phonon Data](https://arxiv.org/abs/2504.21245)**: M. Ohnishi et al., npj Comp. Maters. (2026), arXiv:2504.21245 (2025).

If you use the Python scripts for machine learning predictions, please cite the following papers in addition to the above paper:

- **Original Code**: Z. Chen et al., Direct Prediction of Phonon Density of States With Euclidean Neural Networks, Adv. Sci. 8, 2004214 (2021). 
[Journal](https://onlinelibrary.wiley.com/doi/10.1002/advs.202004214), 
[GitHub1](https://github.com/zhantaochen/phonondos_e3nn), 
[GitHub2](https://github.com/ninarina12/phononDoS_tutorial)

- **E(3)nn**: M. Geiger and T. Smidt, e3nn: Euclidean Neural Networks, arXiv:2207.09453 (2022).

