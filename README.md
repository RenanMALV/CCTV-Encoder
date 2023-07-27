# CCTV-Encoder
## Encoding and decoding algorithm for raw surveillance videos using SVD/PCA

This project uses the Robust PCA algorithm.
For the Robust PCA's optimization problem, we can use the [Primary Component Pursuit Algorithm](https://arxiv.org/pdf/0912.3599.pdf) achieved by the [The Augmented Lagrange Multiplier Method](https://arxiv.org/pdf/1009.5055.pdf), decomposing a video in the form of a matrix into a sum of a low-rank and a sparse matrix.

For the SVD algorithm, We'll use [Facebook's Fast Randomized PCA library](https://github.com/facebookarchive/fbpca).

## Check out the [project's report doc (Portuguese)](https://github.com/RenanMALV/CCTV-Encoder/blob/main/Report_pt.pdf)

## Requirements
  - NumPy >= 2.0.0
  - SciPy >= 1.11.1
  - MoviePy >= 1.0.3
  - fbpca >= 1.0

## Example

```python
"""_summary_
	Demonstrating the usability of the lib
"""
from cctv_encoder import Encoder, Decoder

# Encode process

WORKDIR = "/workspaces/CCTV-Encoder/test_Files/CCTV/"

my_enc = Encoder(WORKDIR)

my_enc.set_precision(1e-2)
my_enc.set_quality(1)
my_enc.set_PCP_n_iter(7)

my_enc.encode(video_path = "CCTV_video.mp4")

# Decode process

decoder = Decoder(WORKDIR)

decoder.decode(encoded_file_path = WORKDIR + "CCTV_video.npz",
               out_file_name = "CCTV_video_foreground_decoded.mp4",
               composition = "foreground")

decoder.decode(encoded_file_path = WORKDIR + "CCTV_video.npz",
               out_file_name = "CCTV_video_background_decoded.mp4",
               composition = "background")

decoder.decode(encoded_file_path = WORKDIR + "CCTV_video.npz",
               out_file_name = "CCTV_video_decoded.mp4",
               composition = "both")
```
