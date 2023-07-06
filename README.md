# CCTV-Encoder
## Encoding and decoding algorithm for raw surveillance videos using SVD/PCA

This project uses the Robust PCA algorithm.
For the Robust PCA's optimization problem, we can use the [Primary Component Pursuit Algorithm](https://arxiv.org/pdf/0912.3599.pdf) achieved by the [The Augmented Lagrange Multiplier Method](https://arxiv.org/pdf/1009.5055.pdf), decomposing a video in the form of a matrix into a sum of a low-rank and a sparse matrix.

For the SVD algorithm, We'll use [Facebook's Fast Randomized PCA library](https://github.com/facebookarchive/fbpca).

## Check out the [project's report doc (TODO: Create report)](https://workinprogress.no/dynamic/upload/bilder/Work-In-Progress.png)

## Requirements
  - NumPy >= 2.0.0
  - SciPy >= 1.11.1
  - MoviePy >= 1.0.3
  - fbpca >= 1.0

## Examples
