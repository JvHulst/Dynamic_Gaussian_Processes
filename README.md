# Estimation_of_Dynamic_Gaussian_Processes
Code accompanying the paper entitled 'Estimation of Dynamic Gaussian Processes'

Paper available at https://heemels.tue.nl/content/papers/HulZui_CDC23a.pdf

Please use the following BibTeX citation when citing this work:
@INPROCEEDINGS{HulZui_CDC23a,
AUTHOR = {J. van Hulst and R. van Zuijlen and D. Antunes and W.P.M.H. Heemels},
TITLE = {Estimation of Dynamic Gaussian Processes},
BOOKTITLE = {IEEE Conference on Decision and Control (CDC) 2023, Singapore },
MONTH = {},
YEAR = {2023},
PAGES = {},
}


To use the method described in the paper, select the basis function structure and the DGP system functions in 'Dynamic_Gaussian_Process_Main'.

Then choose to run the estimator, given a dataset that is generated from one of the following options;
1) A simulated approximate DGP;
2) A solution to the heat equation PDE;
3) An arbitrary dataset.

