# Library for decoding and encoding surveillance camera videos using SVD
# with Principal Component Pursuit

"""_summary_
    Utility functions to perform video encoding and decoding process
"""

import fbpca
import numpy as np

# ==============================================================================
# Auxiliary functions for video transformation
# Necessary for the library's operation
# ==============================================================================

# Get the number of frames in the video


def get_n_frames(video):
    """
    The get_n_frames function takes in a video and returns the number of frames in that video.
    It does this by iterating through all the frames, counting them as it goes.
    
    :param video: video file clip
    :return: The number of frames in a video
    """
    counter = 0
    frames = video.iter_frames()
    for _ in frames:
        counter += 1
    return counter

# Function to convert a matrix with RGB values to a matrix with grayscale values


def rgb_to_gray(rgb):
    """
    The rgb_to_gray function takes in an RGB image and returns a grayscale version of the image.
    
    :param rgb: Store the image data
    :return: A gray scale image
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# Function to convert a matrix with grayscale values to RGB format, while still being grayscale


def gray_to_3gray(gray):
    """
    The gray_to_3gray function takes a single channel image and converts it to a 3-channel image.
    Useful for on memmory reproducing a gray scale video as rgb.
    
    :param gray: Store the image in grayscale
    :return: A 3 channel image with the same value in each channel
    """
    img = np.zeros((gray.shape[0], gray.shape[1], 3))
    for row in range(gray.shape[0]):
        for col in range(gray.shape[1]):
            img[row, col, 0] = gray[row, col]
            img[row, col, 1] = gray[row, col]
            img[row, col, 2] = gray[row, col]
    return img

# Convert the video into a matrix structure, as defined in the notebook's introduction
def video_to_matrix(video):
    """
    The video_to_matrix function takes a video and returns a matrix of the frames.
    The function iterates through each frame in the video, converts it to grayscale,
    stores it on a Nx1 dimension vector and appends it to the video array.
    The array is then returned to the caller.
    
    :param video: Specify the video file to be converted into a matrix
    :return: The matrix with the frames disposed in each column
    """
    frames = video.iter_frames()
    return np.vstack([rgb_to_gray(frame).flatten() for frame in frames]).T

# Convert the matrix back to an image given its dimensions and a frame number
def matrix_to_image(A, height, width, frame=0):
    """
    The matrix_to_image function takes a matrix A, height, width and frame number as input.
    It returns the image corresponding to the frame number in A.
    
    :param A: Store the video data, height and width are used to reshape the
    frame vector into a image matrix
    :param height: Set the height of the image
    :param width: Set the width of the image
    :param frame: Select which frame of the video we want to display
    :return: A single image from the matrix's frame
    """
    return np.reshape(A[:, frame], (height, width))

# Perform the singular value decomposition (SVD) on the matrix
def _svd(M, rank):
    """
    The _svd function is a wrapper for the fbpca.pca function, which performs
    a truncated SVD on an input matrix M. The rank parameter specifies the number of
    singular values to keep in the decomposition; if this value is greater than 
    the minimum dimension of M, then it will be set to that minimum dimension instead. 
    The output from _svd is a tuple containing three matrices: U, sigma and Vt.
    
    :param M: Pass in the matrix that we want to decompose
    :param rank: Determine the number of singular values to be returned
    :return: A call to  the facebook pca method
    """
    return fbpca.pca(M, k=min(rank, np.min(M.shape)), raw=True)

# Reconstruct the matrix using the singular value decomposition (SVD)
# with a specified rank and minimum singular value
def svd_reconstruct(M, rank, min_sv):
    """
    The svd_reconstruct function takes a matrix M, the rank of the SVD decomposition,
    and a minimum singular value.
    It returns an approximation to L using only those singular values greater than min_sv.
    
    :param M: Pass in the matrix to be decomposed
    :param rank: Specify the number of singular values to use in the reconstruction
    :param min_sv: Remove the singular values that are too small
    :return: A tuple of two elements:
    An estimate for the low rank matrix
    The singular values greaater than min_sv
    """
    u, s, v = _svd(M, rank)
    s -= min_sv
    nnz = (s > 0).sum()
    return u[:, :nnz] @ np.diag(s[:nnz]) @ v[:nnz], nnz

# Check if the algorithm has converged by comparing the error with the given tolerance
def converged(tol, Z, d_norm):
    """
    The converged function checks to see if the algorithm has converged.
    
    :param tol: The tolerance for convergence. Determines when to stop the algorithm
    :param Z: The current estimate of the solution matrix, Z.
    :param d_norm: The norm of D, used in calculating error.
    :return: A boolean value about convergence.
    """
    err = np.linalg.norm(Z, 'fro') / d_norm
    print('error:', err)
    return err < tol

# Shrink the matrix using a given threshold (tau)
def shrink(M, tau):
    """
    Shrink the cell values of M by tau. the values shrinked can not change signal.
    The shrink function takes a matrix M and a threshold tau as input.
    It then computes the absolute value of M, subtracts tau from it, 
    and returns the sign of M times the maximum between 0 and S.
    This is equivalent to soft-thresholding.
    
    :param M: Pass in the matrix that we want to shrink
    :param tau: Determine the threshold of the shrinkage function
    :return: A matrix with the same dimensions as m, but
    with each element replaced by its sign times max(abs(element) - tau, 0)
    """
    S = np.abs(M) - tau
    return np.sign(M) * np.where(S > 0, S, 0)

# Compute the operator norm of the matrix


def norm_op(M):
    """
    The norm_op function takes a matrix M as input and returns the largest singular value of M
    as the normalization operator.
    
    :param M: The matrix whose largest singular value is to be computed.
    :return: The largest singular value of the matrix M
    """
    return _svd(M, 1)[1][0]
