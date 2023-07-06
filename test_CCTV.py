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
