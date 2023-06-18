# Library for decoding and encoding surveillance camera videos using SVD
# with Principal Component Pursuit

"""_summary_
    Decoder of encoded data
"""

import os

import moviepy.editor as mpEdit
import numpy as np
from scipy.sparse import csr_matrix

from cctv_encoder.util_functions import matrix_to_image, gray_to_3gray


class Decoder:
    """_summary_
    Class for decoding the encoded file
    """

    def __init__(self, work_directory, encoded_file_path=""):
        """
        The __init__ function is the constructor for a class.
        It is called when an object of that class is created.

        :param self: Represent the instance of the class
        :param work_directory: Set the working directory of the class
        :param encoded_file_path: Store the path of the encoded file
        :return: Nothing
        """
        self.encoded_file_path = encoded_file_path
        self.work_directory = work_directory

        if encoded_file_path != "":
            self.decode(encoded_file_path)

    def load_encoded_file(self, encoded_file_path):
        """
        The load_encoded_file function takes in a path to an encoded file and
        returns the csr_m, background, and meta_data.
        The function loads the encoded data from the specified file using np.load(). 
        It then creates a list of keys from that loaded data (keys = list(encoded_data.keys())). 
        Then it assigns each key to its respective variable: 
        csr_m = encoded_data[keys[0]], background = encoded_data[keys[2]],
        meta_data = encoded_data[key3]. 

        Parameters:
            -encoded_file_

        :param self: Represent the instance of the class
        :param encoded_file_path: Specify the path of the encoded file
        :return: A tuple of three items:
        :doc-author: Trelent
        """
        encoded_data = np.load(encoded_file_path, allow_pickle=True)
        keys = list(encoded_data.keys())
        csr_m = encoded_data[keys[0]]
        background = encoded_data[keys[1]]
        meta_data = encoded_data[keys[2]]
        return csr_m, background, meta_data

    def decompress(self, csr_m, background):
        """
        The decompress function takes a compressed matrix and the background value,
        and returns the original matrix. The decompress function is used to reconstruct
        the original data from its compressed form.

        :param self: Represent the instance of the class
        :param csr_m: Store the compressed sparse row matrix
        :param background: Add the background to the matrix s
        :return: A matrix of the same shape as a, but with values scaled to be between 0 and 1
        :doc-author: Trelent
        """
        S = csr_matrix((csr_m[0], csr_m[1], csr_m[2])).todense()
        A_hat = (np.ones(S.shape).T * background).T
        A_hat += S
        return A_hat

    def matrix_to_video(self, A_hat, width, height, fps):
        """
        The matrix_to_video function takes in a matrix of shape (n, m) and returns an mp4 video
        with n frames.

        :param self: Represent the instance of the class
        :param A_hat: Get the frames from the matrix
        :param width: Set the width of the video
        :param height: Specify the height of the image
        :param fps: Determine the frame rate of the video
        :return: A video clip
        :doc-author: Trelent
        """
        frames = []
        for frame_idx in range(A_hat.shape[1]):
            frame = matrix_to_image(A_hat, height, width, frame=frame_idx)
            frame = gray_to_3gray(frame.astype(np.uint8))
            frames.append(frame)
        return mpEdit.ImageSequenceClip(frames, fps=fps)

    def decode(self, encoded_file_path):
        """
        The decode function takes in an encoded file path and returns a decoded video file.
        The function first loads the encoded data from the given path, then decompresses it using 
        the decompress function.
        Finally, it converts this matrix into a video and writes it to disk.

        :param self: Bind the method to an object
        :param encoded_file_path: Load the encoded file
        :return: The path to the decoded video
        :doc-author: Trelent
        """
        self.encoded_file_path = encoded_file_path
        csr_m, background, meta_data = self.load_encoded_file(encoded_file_path)
        self.width = meta_data[0]
        self.height = meta_data[1]
        self.fps = meta_data[2]
        A_hat = self.decompress(csr_m, background)
        video = self.matrix_to_video(A_hat, self.width, self.height, self.fps)

        video_name = self.encoded_file_path.split(
                sep="/")[-1].split(sep=".")[0]
        os.makedirs(self.work_directory + "/", exist_ok=True)
        output_file_path = self.work_directory + "/" + video_name + "_decoded.mp4"
        video.write_videofile(output_file_path, codec="libx264")

        return output_file_path