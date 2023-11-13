# adversarial_attacks_and_sparse_representations

This repo allows to recreate the experiments and plots from the paper "[On the Relationship Between Universal Adversarial Attacks and Sparse Representations](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10043630)".

## Recreate the figures
All the run files are configured with valid defaults and documented with the available flagged changes.

Fig. 1, 6: 
```bash
./evaluate_DA_on_classification.py
```

Fig. 2:
```bash
./scatter_plot_attack_delta_sc.py
```

Fig. 3, 4:
```bash
./load_corrs_and_plot.py
```

Fig. 5:
```bash
./hist_delta_energy.py
```

Fig. 7:
```bash
evaluate_DA_on_classification.py
```

## Train sparse coding models
To train a sparse coder, please run 
```bash
./train_sparse_coding_models/train_sparse_coder.py
```

To train a linear classifier over the sparse code, please run 
```bash
./train_sparse_coding_models/train_linear_classifier_over_sparse_code.py
```
## Don't forget to cite!

If you find this repo useful or mention us in your work, please don't forget to cite us.

@article{weitzner2023relationship,
  title={On the Relationship Between Universal Adversarial Attacks and Sparse Representations},
  author={Weitzner, Dana and Giryes, Raja},
  journal={IEEE Open Journal of Signal Processing},
  volume={4},
  pages={99--107},
  year={2023},
  publisher={IEEE}
}
