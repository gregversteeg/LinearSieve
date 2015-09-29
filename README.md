# The Linear, Gaussian Information Sieve

While this version of the sieve makes some strong assumptions: latent factors are linear functions of the data and we assume that the form of marginal distributions are Gaussian, the algorithm is supefast, stable, and scalable. 

The weights for each latent factor are found incrementally by optimizing a nonlinear objective, max TC(X;Y).  
