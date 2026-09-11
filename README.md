# GAD.jl
GAD.jl is a Julia package for computing Generalized Additive Decomposition (GAD) of symmetric tensors.
GAD.jl is distributed under GNU GPL v3.


## Installation

To install GAD.jl, start Julia and run:

```julia
using Pkg

Pkg.add(url="https://github.com/AlgebraicGeometricModeling/TensorDec.jl")
Pkg.add(url="https://github.com/enricabarrilli/GAD.jl")
```

Once the installation is complete, load GAD.jl with:

```julia
using GAD
```

## Example

The following example illustrates how to compute a Generalized Additive Decomposition of a homogeneous polynomial using GAD.jl.

First, define the polynomial variables and construct the polynomial to be decomposed:

```julia
using GAD
using DynamicPolynomials

X = @polyvar x0 x1 x2

d = 5
F = 0.5 * (x0 + x1 + x2)^d + (x0 + x1) * (x0 - x2)^(d - 1)
```

The Generalized Additive Decomposition is then computed with:

```julia
W, L, mu = GAD.gad_decompose(F)
```

The algorithm returns three vectors:

- `W`, containing the polynomials \(\omega_i\);
- `L`, containing the linear forms \(\ell_i\);
- `mu`, containing the multiplicities of the corresponding points.


## Dependencies
* TensorDec
* MultivariateSeries
* LinearAlgebra
* DynamicPolynomials
* Clustering
* Distances
