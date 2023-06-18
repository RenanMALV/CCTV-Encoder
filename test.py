"""_summary_
	Demonstrating the usability of the lib
"""
from cctv_encoder import Encoder, Decoder

# Encode process

WORKDIR = "/test_Files"

my_enc = Encoder(WORKDIR)

my_enc.set_precision(1e-2)
my_enc.set_quality(1)
my_enc.set_PCP_n_iter(8)

my_enc.encode(video_path = "video.mp4")

# Decode process

decoder = Decoder(WORKDIR)
decoder.decode(encoded_file_path = WORKDIR + "/video.npz")
