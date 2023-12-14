# Estimation_of_Dynamic_Gaussian_Processes

## Code accompanying the paper entitled 'Estimation of Dynamic Gaussian Processes'

**Paper available at https://heemels.tue.nl/content/papers/HulZui_CDC23a.pdf**


Any questions, inquirires or issues can be communicated to j.s.v.hulst@tue.nl



Please use the following BibTeX citation when citing this work:

```
@INPROCEEDINGS{Hulst2023,
AUTHOR = {Jilles van Hulst and Roy van Zuijlen and Duarte Antunes and W.P.M.H. Heemels},
TITLE = {Estimation of Dynamic Gaussian Processes},
BOOKTITLE = {2023 IEEE 62nd Conference on Decision and Control (CDC)}
MONTH = {December},
YEAR = {2023},
}
```

To use the method described in the paper, select the basis function structure and the DGP system functions in 'Dynamic_Gaussian_Process_Main'.

Then choose what dataset the estimator uses from one of the following options;
1) A simulated approximate DGP;
2) A solution to the heat equation PDE, discretized in time;
3) An arbitrary dataset.

