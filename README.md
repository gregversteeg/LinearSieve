# The Linear, Gaussian Information Sieve

While this version of the sieve makes some strong assumptions: latent factors are linear functions of the data and we assume that the form of marginal distributions are Gaussian, the algorithm is supefast, stable, and scalable. 

The weights for each latent factor are found incrementally by optimizing a nonlinear objective, max TC(X;Y).  
Therefore, we now specify just one number: the maximum number of latent factors to use, n_hidden, and the structure is
learned automatically.

In the tests folder, try:
```
python test_weak_correlations.py
python test_faces.py
python vis_sieve.py tests/data/test_big5.csv --n_hidden=5 -v --no_row_names -o big5
python vis_corex.py tests/data/adni_blood.csv --n_hidden=30 --missing=-1e6 -v -o adni
```
Each of these examples generates pairwise plots of relationships and a graph. 

Note that missing values are imputed in vis_sieve beforehand. If you are using the command line API,
you should impute missing values manually (as the mean within each column). 