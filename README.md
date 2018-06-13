# HaskellGBM

A Haskell wrapper around the [Microsoft LightGBM
library](https://github.com/Microsoft/LightGBM) for machine learning
uses.  The emphasis here is on using Haskell types to help ensure that
the hyperparameter settings chosen by the user are coherent and
in-bounds at all times.

This software is not on Hackage yet - in the meantime you can find
Haddock API documentation
[here](https://dpkatz.github.io/haddocks/HaskellGBM-0.1.0.0/index.html)
and a tutorial blog post
[here](https://dpkatz.github.io/posts/using-lightgbm-from-haskell/).

__N.B. This package is still under heavy development including API
changes and should not be used in production code.  Contributions,
suggestions, and PRs are welcome.__

## Installation

  - Install the [LightGBM library](http://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)
  - Add the location of the 'lightgbm' executable to you PATH
  - Install the Haskell ['stack' build tool](https://docs.haskellstack.org/en/latest/install_and_upgrade/)
  - Clone the HaskellGBM source code:
``` shell
$ git clone https://github.com/dpkatz/HaskellGBM.git
```
  - Build HaskellGBM

``` shell
$ stack setup
$ stack build
$ stack test
```

    
