# gp_constr_modified
Modification of the original [gp_constr](https://github.com/cagrell/gp_constr) library by C. Agrell, adding the possibility of adding concavity/convexity constraints to the Gaussian process. Please cite the [original paper](https://arxiv.org/abs/1901.03134) if this repository has been helpful.

The code in this repository was used in an upcoming paper on constrained Gaussian processes for regression in renewable energy applications.

Obs.: no crossed second derivative constraints allowed, only d^2F/dx^2.

Obs.: prior distribution mean must be necessarily equal to zero
