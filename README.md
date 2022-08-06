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
```
@article{Xiong_2022,
  title = {A Model {RRNet} for Spectral Information Exploitation and {LAMOST} Medium-resolution Spectrum Parameter Estimation},
  author = {Shengchun Xiong and Xiangru Li and Caixiu Liao},
  journal = {The Astrophysical Journal Supplement Series},
  doi = {10.3847/1538-4365/ac76c6},
  url = {https://doi.org/10.3847/1538-4365/ac76c6},
  year = 2022,
  month = {aug},
  publisher = {American Astronomical Society},
  volume = {261},
  number = {2},
  pages = {36},
}
```
```
@article{xiong2022rrnet,
  title={A Model RRNet for Spectral Information Exploitation and LAMOST Medium-resolution Spectrum Parameter Estimation},
  author={Xiong, Shengchun and Li, Xiangru and Liao, Caixiu},
  journal={arXiv preprint arXiv:2205.15490},
  year={2022}
}
```
