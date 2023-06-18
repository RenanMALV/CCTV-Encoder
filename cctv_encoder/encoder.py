# Library for decoding and encoding surveillance camera videos using SVD
# with Principal Component Pursuit

"""_summary_
	Module that encodes a CCTV video file
"""

import os

import moviepy.editor as mpEdit
import numpy as np
from scipy.sparse import csr_matrix
from cctv_encoder.util_functions import converged, svd_reconstruct, video_to_matrix, shrink, norm_op

# ==============================================================================
# Encoder class for video encoding
# The class does not manage memory, so when working with large files,
# make sure to split them into appropriate subclips
# ==============================================================================


class Encoder:
    """_summary_
      Encoder of surveillance video file
    """
    # Constructor for the Encoder class
    # Args:
    # the working directory (to return the encoded file);
    # the path of the video that will be optionally encoded once the encoder is instantiated;
    # number of PCP iterations;
    # the tol value in decimal order for creating the sparse matrix,
    # quality of details that will be captured in the foreground
    # the smaller this value, the sparse the array. value >= 1;
    def __init__(self, work_directory, video_path="", n_iter=8, tol=1e-2, quality=1):
        """
		The __init__ function is called when the class is instantiated.
		It sets up the instance of the class, and defines all of its attributes.
		
		
		:param self: Represent the instance of the class
		:param work_directory: Set the directory where the video is stored
		:param video_path: Specify the path of the video to be encoded
		:param n_iter: Determine the number of iterations to run through the algorithm
		:param tol: Determine the tolerance for the convergence of the algorithm
		:param quality: Determine the quality of the video
		:return: Nothing
		:doc-author: Trelent
		"""
        self.work_directory = work_directory
        self.video_path = video_path
        self.n_iter = n_iter
        self.tol = tol
        self.quality = quality

        if video_path != "":
            self.encode(video_path)

    # Compress the sparse matrix
    def compress(self, S):
        """
		The compress function takes a sparse matrix S and returns the compressed version of it.
		The compression is done by converting the sparse matrix to a csr_matrix, which is 
		more efficient in terms of storage.
		
		:param self: Represent the instance of the class
		:param S: Store the sparse matrix
		:return: A compressed sparse row matrix
		:doc-author: Trelent
		"""
        return csr_matrix(S)

    # Save the compressed object to a file
    def save_compressed_file(self, csr_m, background):
        """
		The save_compressed_file function takes in a csr_matrix, and a background image.
		It then converts the csr_matrix into three arrays: data, indices, and indptr.
		The function then calculates the total size of these arrays in KiB (kilobytes). 
		Then it creates an array called compressed_foreground that contains all three of these arrays. 
		Finally it saves this array as well as the background image to a .npz file.
		
		:param self: Access the instance attributes and methods
		:param csr_m: Store the sparse matrix
		:param background: Store the background image
		:return: The compressed_foreground, background and meta_data
		:doc-author: Trelent
		"""
        csr_data = np.array(csr_m.data).astype(np.int16)
        csr_index = np.array(csr_m.indices)
        csr_idx_ptr = np.array(csr_m.indptr)
        total_size_KiB = (csr_data.nbytes + csr_index.nbytes +
                        csr_idx_ptr.nbytes + background.nbytes) / 1024
        print("Compressed data size:", total_size_KiB, "KiB")

        compressed_foreground = np.array(
            [csr_data, csr_index, csr_idx_ptr], dtype=object)
        meta_data = np.array([self.width, self.height, self.fps])

        video_name = self.video_path.split(sep="/")[-1].split(sep=".")[0]
        os.makedirs(self.work_directory + "/", exist_ok=True)
        np.savez(self.work_directory + "/" + video_name + ".npz",
                 compressed_foreground, background, meta_data)

    # Set the number of iterations for the PCP algorithm
    def set_PCP_n_iter(self, n_iter):
        """
		The set_PCP_n_iter function sets the number of iterations for the PCP algorithm.
				
		:param self: Represent the instance of the class
		:param n_iter: Set the number of iterations for the pcpn algorithm
		:return: The number of iterations
		:doc-author: Trelent
		"""
        self.n_iter = n_iter

    # Set the precision of the optimisation algorithm for creating the sparse matrix
    def set_precision(self, tol):
        """
		The set_precision function sets the tolerance for the root finding algorithm.
		The default value is 1e-6, but this can be changed by calling set_precision(tol)
		where tol is a float representing the desired tolerance.
		
		:param self: Represent the instance of the class
		:param tol: Set the precision of the output
		:return: The current value of the tolerance
		:doc-author: Trelent
		"""
        self.tol = tol

    # Set the quality of the details captured in the foreground
    # the smaller this value, sparser is the foreground array. value >= 1;
    # represents the K value in the PCP algorithm
    def set_quality(self, quality):
        """
		The set_quality function sets the quality of a song.
		Args:
		quality (int): The quality of details on foreground. 
		
		:param self: Represent the instance of the class
		:param quality: Set the quality of the foreground
		:return: Nothing
		:doc-author: Trelent
		"""
        self.quality = quality

    # ============================================================================
    # PCP Algorithm
    # ============================================================================

    def pcp(self, X, maxiter=10, k=1):
        """
		The pcp function performs the Principal Component Pursuit algorithm on a given matrix X.
		
		:param self: Refer to the instance of the class
		:param X: Represent the input matrix
		:param maxiter: Set the maximum number of iterations to perform
		:param k: Set the initial value of mu
		:return: A tuple of two matrices: the low-rank matrix and the sparse matrix
		:doc-author: Trelent
		"""
        m, n = X.shape
        trans = m < n
        if trans:
            X = X.T
            m, n = X.shape

        lamda = 1 / np.sqrt(m)
        op_norm = norm_op(X)
        Y = np.copy(X) / max(op_norm, np.linalg.norm(X, np.inf) / lamda)
        mu = k * 1.25 / op_norm
        mu_bar = mu * 1e7
        rho = k * 1.5

        d_norm = np.linalg.norm(X, 'fro')
        L = np.zeros_like(X)
        sv = 1

        for _ in range(maxiter):
            print("rank sv:", sv)
            X2 = X + Y / mu

            # Update the estimate of the Sparse Matrix
			# by "shrinking/truncating": original - low-rank
            S = shrink(X2 - L, lamda / mu)

            # Update the estimate of the Low-rank Matrix by performing
			# truncated SVD of rank sv & reconstructing.
            # The count of singular values > 1/mu is returned as svp
            L, svp = svd_reconstruct(X2 - S, sv, 1 / mu)

            # If svp < sv, you are already calculating enough singular values.
            # If not, add 20% (in this case 240) to sv
            sv = svp + (1 if svp < sv else round(0.05 * n))

            # Residual
            Z = X - L - S
            Y += mu * Z
            mu *= rho

            if m > mu_bar:
                m = mu_bar
            if converged(self.tol, Z, d_norm):
                break

        if trans:
            L = L.T
            S = S.T
        return L, S

    # Video encoding function
    def encode(self, video_path):
        """
		The encode function takes a video file and compresses it using the PCP algorithm.
		The function returns two matrices, L and S, which are the low-rank matrix 
		(background) and sparse matrix (foreground), respectively.
		
		:param self: Access the attributes and methods of the class
		:param video_path: Specify the path of the video to be encoded
		:return: Two matrices l and foreground
		:doc-author: Trelent
		"""
        self.video_path = video_path
        raw_video = mpEdit.VideoFileClip(self.video_path)
        raw_video = raw_video.subclip(0, int(raw_video.duration))
        self.width = int(raw_video.size[0])
        self.height = int(raw_video.size[1])
        self.fps = int(raw_video.fps)

        # Create matrix A
        A = video_to_matrix(raw_video)

        L, S = self.pcp(A, maxiter=self.n_iter, k=self.quality)
        foreground = S.astype(np.int16)
        background = L[:, 0].astype(np.int16)

        print("Sparsity:", (foreground.size - np.count_nonzero(foreground))
              * 100 / foreground.size, "%")

        # Save the compressed matrix to a file
        self.save_compressed_file(self.compress(foreground), background)
