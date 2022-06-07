# A Model RRNet for Spectral Information Exploitation and LAMOST Medium-resolution Spectrum Parameter Estimation

This repo contains the code, trained models and LAMOST-RRNet catalog for our paper *A Model RRNet for Spectral Information Exploitation and LAMOST Medium-resolution Spectrum Parameter Estimation*.


## Requirements
- PyTorch
- numpy
- pandas
- matplotlib
- sklearn
- jupyter

## Experimental data
-  Download website: <https://doi.org/10.12149/101112> and <https://github.com/Chan-0312/RRNet/releases>.
- Please put `test_flux.pkl`, `valid_flux.pkl` and `train_flux.pkl` in the `data/refer_set/` directory and `DR7MRS_RRNet_parametes.csv` in the `data/` directory.

## Usage

- Training a new model:
```shell
python train.py
```

- Test:
```shell
python predict.py
```

- More examples can be found in the `jupyter/` directory.


## Citation

- If you found this code useful please cite our paper: 
