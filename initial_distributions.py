import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



def build_2d_gaussian(mean=(0,0), covar=0, random_seed=1000):
    """
    Build 2d gaussian pdf.
    :param mean: 2-tuple of the centre of the distribution
    :param covar: single covariance value to construct the covariance matrix from.
    :param random_seed:
    :return: 2d distribution, X mesh, Y mesh (mesh = coordinates)
    """

    # Initializing the covariance matrix
    cov = np.array([[1, covar], [covar, 1]])

    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix
    distr = multivariate_normal(cov = cov, mean = mean,
                                seed = random_seed)

    # Generating a meshgrid complacent with
    # the 3-sigma boundary
    mean_1, mean_2 = mean[0], mean[1]
    sigma_1, sigma_2 = cov[0,0], cov[1,1]

    x = np.linspace(-3*sigma_1, 3*sigma_1, num=100)
    y = np.linspace(-3*sigma_2, 3*sigma_2, num=100)
    X, Y = np.meshgrid(x,y)

    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])

    return pdf, X,Y

def valley_distr(mean=(0,0), covar=0, random_seed=1000):
    """
    Build 2d U-shape distribution.
    :param mean: 2-tuple of the centre of the distribution
    :param covar: single covariance value to construct the covariance matrix from.
    :param random_seed:
    :return:
    """
    pdf, X, Y = build_2d_gaussian(mean=mean,covar=covar, random_seed=random_seed)

    gauss_max = np.max(pdf)
    return -1*pdf + gauss_max, X, Y



def plot_distribution(pdf, X, Y, args):
    """
    Plot 2D initial distributions.
    :param pdf: Distribution of values on the 2d Grid
    :param X:  X coordinate grid
    :param Y: Y coordinate grid
    :param args: list of arguments to the function
    :return:
    """
    fig = plt.figure()
    plt.subplot(1,3,1)
    plt.contourf(X, Y, pdf, cmap='viridis')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f'Arguments are = {args}')


    ax = fig.add_subplot(132, projection = '3d')
    ax.plot_surface(X, Y, pdf, cmap = 'viridis')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f'Covariance between x1 and x2 = {args}')
    plt.tight_layout()
    plt.show()
