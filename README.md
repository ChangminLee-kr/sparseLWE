# Integer_LWE Solver

This repository implements three algorithms for solving the Integer_LWE problem:
- **Our Algorithm**
- **NMW+24**: [The cool and the cruel: separating hard parts of lwe secrets](https://eprint.iacr.org/2024/443.pdf)
- **CCP+24**: [Attacks against the INDCPA-D security of exact FHE schemes]( https://eprint.iacr.org/2024/127.pdf )

## Parameters

Each algorithm takes the following identical parameters as input:
- `n`: Secret dimension
- `m`: Number of samples
- `Hw`: Hamming weight of the secret vector
- `q`: Underlying modulus
- `sigma`: Noise parameter for discrete Gaussian

## How to Use

1. Install Sage.
2. Call the function to run the algorithms as follows:

   ```python
   Integer_LWE_solver(n, Hw, q, m, sigma)
3. Check the results of the algorithms.


