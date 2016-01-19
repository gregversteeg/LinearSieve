# The Linear, Gaussian Information Sieve

While this version of the sieve makes some strong assumptions: 
latent factors are linear functions of the data and we assume that the form of marginal distributions are Gaussian, 
the algorithm is super-fast, stable, and scalable. 

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

Note that missing values are imputed in vis_sieve beforehand. Using the command line API, entries that are np.nan
are treated as missing. 

You can do in-painting experiments by training a model, then mask missing values in a test set with np.nan values, 
then generate y's for these test examples. Finally, use the .predict() method to predict the values of all x's from the 
y's. 
```
out = sieve.Sieve(n_hidden=3).fit(x_train)
y = out.transform(x_test)  # missing values in x_test are set to np.nan
x_predict = out.predict(y)  # all values are predicted, missing and not missing

```