import sys
import numpy as np
sys.path.append("./")


def extract_efd_feature(contour):
    """Function to fit a Fourier series approximating the shape of the contour

    Args:
        contour (N*1*2 numpy array): contour point set

    Returns:
        numpy array: a flattened array as Fourier descriptor
    """
    from pyefd import elliptic_fourier_descriptors
    contour = np.squeeze(contour)
    coeffs = elliptic_fourier_descriptors(contour, order=10, normalize=True)
    
    # return coeffs.flatten()[3:]
    return coeffs[1:, 0].flatten() # only use a_1, a_2, ... a_n


def extract_zernike_moments_feature(im, radius=21, degree=8):
    """Extract zernike moments feature from the target image
        
    Args:
        im (numpy array): a grayscale image
        radius (int, optional): set the region of which the polynomials are defined. The input image is mapped to a disc with radius r, where the center of the image is placed at the origin of the disc. The radius r should technically be set properly to include the entire region of the shape. Defaults to 21.
        degree (int, optional): the degree of the polynomial. Defaults to 8.

    Returns:
        numpy array: a flatten array of zernike moments vector
    """
    import mahotas
    moments = mahotas.features.zernike_moments(im, radius, degree)

    return moments