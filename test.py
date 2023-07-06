"""_summary_
	Demonstrating the usability of the lib
"""
from cctv_encoder import Encoder, Decoder

# Encode process

WORKDIR = "/workspaces/CCTV-Encoder/test_Files/BallBounce/"

my_enc = Encoder(WORKDIR)

my_enc.set_precision(1e-2)
my_enc.set_quality(1)
my_enc.set_PCP_n_iter(7)

my_enc.encode(video_path = "BallBounce_30sec_(144p).mp4")

# Decode process

decoder = Decoder(WORKDIR)

decoder.decode(encoded_file_path = WORKDIR + "BallBounce_30sec_(144p).npz",
               out_file_name = "BallBounce_foreground_decoded.mp4",
               composition = "foreground")
