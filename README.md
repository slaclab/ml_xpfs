# A machine learning photon detection algorithm for coherent X-ray ultrafast fluctuation analysis

U-net convolutional neural network for the analysis of X-ray Photon Fluctuation Spectroscopy experiments (XPFS). The data and code here corresponds to "A machine learning photon detection algorithm for coherent X-ray ultrafast fluctuation analysis" (Structural Dynamics, 2022). 

<img width="1448" alt="Screen Shot 2022-09-26 at 4 33 19 PM" src="https://user-images.githubusercontent.com/39596225/192398754-1d2a5e2e-2142-4c9d-86e7-78da35966fdb.png">

## Requirements

For reproducibility we encourage the use of containerization. Please pull the appropriate Docker container.

```
docker pull slaclab/slac-ml:20211101.0
```

## Data

All data presented in this repository can be accessed at: 

```
https://zenodo.org/record/6643622#.YzIqnuzMKUU
```

To train models, please place the extracted XPFS_data folder in the main ml_xpfs folder. 

## Training, Testing and Uncertainty Quantification 

Please see relevant jupyter notebooks in the src directory for examples on how to perform training, testining and uncertainty quantification. For models, please see the models folder. 

## Citations 

If you find this repository or paper useful, please consider citing the following. 

- data: 10.5281/zenodo.6643621
- paper: Chitturi, Sathya R., Nicolas G. Burdet, Youssef Nashed, Daniel Ratner, Aashwin Mishra, T. J. Lane, Matthew Seaberg et al. "A machine learning photon detection algorithm for coherent X-ray ultrafast fluctuation analysis." Structural Dynamics (2022).
