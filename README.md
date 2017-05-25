# Extract common information with the linear sieve

This code is based on the following reference: 
- [Greg Ver Steeg, Shuyang Gao, Kyle Reing, and Aram Galstyan. "Sifting Common Information from Many Variables"](https://arxiv.org/abs/1606.02307)

This work in turn builds on the theoretical results here: 
- [Greg Ver Steeg and Aram Galstyan. "The Information Sieve", ICML 2016.](http://arxiv.org/abs/1507.02284)

While this version of the sieve makes some strong assumptions: 
latent factors are linear functions of the data and we assume that the form of marginal distributions are Gaussian, 
the algorithm is super-fast, stable, and scalable. As discussed in the paper, you are free to gaussianize the marginals
of each observed variable yourself ahead of time, perhaps using [this code](http://github.com/gregversteeg/gaussianize). 

The weights for each latent factor are found incrementally by optimizing a nonlinear objective, max TC(X;Y).  
Therefore, we now specify just one number: the maximum number of latent factors to use, n_hidden, and the structure is
learned automatically. As we add new latent factors, more and more common information is extracted until eventually
TC(X|Y) = 0. In other words, the data is independent conditioned on Y. 

###Dependencies

Linear sieve requires numpy. If you use OS X, I recommend installing the [Scipy Superpack](http://fonnesbeck.github.io/ScipySuperpack/).

The visualization capabilities in vis_sieve.py require other packages: 
* matplotlib - Already in scipy superpack.
* [networkx](http://networkx.github.io)  - A network manipulation library. 
* sklearn - Already in scipy superpack and only required for visualizations. 
* [graphviz](http://www.graphviz.org) (Optional, for compiling produced .dot files into pretty graphs. The command line 
tools are called from vis_sieve. Graphviz should be compiled with the triangulation library for best visual results).

###Install

To install, download using [this link](https://github.com/gregversteeg/LinearSieve/archive/master.zip) 
or clone the project by executing this command in your target directory:
```
git clone https://github.com/gregversteeg/LinearSieve.git
```
Use *git pull* to get updates. The code is under development. 
Please contact me about issues. 


###Example tests and command line interface
We generally try to follow the sklearn coding style using .fit(), .transform(), and .predict() methods. 

Try the following tests from the project directory:
```
python tests/test_weak_correlations.py
python tests/test_faces.py
python vis_sieve.py tests/data/test_big5.csv --n_hidden=5 -v --no_row_names -o big5
python vis_sieve.py tests/data/adni_blood.csv --n_hidden=30 --missing=-1e6 -v -o adni
```
Vis_sieve.py automatically generates different plots.  In the relationships folder, for each latent factor you will 
see pairwise plots of observed variables related to that factor. The color of the dot corresponds to the latent factor.
The graphs folder will show the hierarchical relationships. The file tree_sfdp_w_splines.pdf is usually the best. 

Note that missing values are imputed in vis_sieve beforehand. Using the command line API, entries that are np.nan
are treated as missing. 

You can do in-painting experiments by training a model, then mask missing values in a test set with np.nan values, 
then generate y's for these test examples. Finally, use the .predict() method to predict the values of all x's from the 
y's. Here's an example of how the python API looks.
```
import linearsieve as sieve
ns, nv = x_train.shape  # x_train is an array with rows for samples and columns for variables.
out = sieve.Sieve(n_hidden=3).fit(x_train)  
y = out.transform(x_test)  # missing values in x_test are set to np.nan
x_predict = out.predict(y)  # all values are predicted, missing and not missing
print out.ws[:,:nv]  # These are the weights for how each latent factor depends on each variable.
```
