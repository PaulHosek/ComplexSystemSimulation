import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import imageio.v3 as iio



def build_2d_gaussian(mean=(0,0), covar=0, random_seed=1000, size = 100):
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
    distr = stats.multivariate_normal(cov = cov, mean = mean,
                                seed = random_seed)

    # Generating a meshgrid complacent with
    # the 3-sigma boundary
    mean_1, mean_2 = mean[0], mean[1]
    sigma_1, sigma_2 = cov[0,0], cov[1,1]

    x = np.linspace(-3*sigma_1, 3*sigma_1, num=size)
    y = np.linspace(-3*sigma_2, 3*sigma_2, num=size)
    X, Y = np.meshgrid(x,y)

    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])

    return pdf, X,Y

def valley_distr(mean=(0,0), covar=0, random_seed=1000,size=100):
    """
    Build 2d U-shape distribution.
    :param mean: 2-tuple of the centre of the distribution
    :param covar: single covariance value to construct the covariance matrix from.
    :param random_seed:
    :param size: side length of grid
    :return:
    """
    pdf, X, Y = build_2d_gaussian(mean=mean,covar=covar, random_seed=random_seed,size=size)

    gauss_max = np.max(pdf)
    return -1*pdf + gauss_max, X, Y


def build_2d_beta(alpha=(2, 2), beta=(2, 2), size=100):
    """
    Build 2D beta pdf.
    :param alpha: 2-tuple of shape parameters (alpha1, alpha2)
    :param beta: 2-tuple of shape parameters (beta1, beta2)
    :param size: Number of points in each dimension of the meshgrid
    :return: 2D distribution, X mesh, Y mesh (mesh = coordinates)
    """

    # Generating a beta bivariate distribution
    # with given shape parameters
    pdf = stats.beta(a=alpha[0], b=beta[0])

    # Generating a meshgrid
    x = np.linspace(0, 1, num=size)
    y = np.linspace(0, 1, num=size)
    X, Y = np.meshgrid(x, y)

    # Generating the density function
    # for each point in the meshgrid
    # pdf = joint.pdf(np.stack([X, Y], axis=-1))

    return pdf, X, Y

def multi_valley(mean1=(0, 0), mean2=(5, 5), cov1=([4, 0], [0, 3]),
                 cov2=([4, 0.9], [0.6, 3]),size=125):

    # Create the first peak distribution
    dist1 = stats.multivariate_normal(mean=mean1, cov=cov1)

    # Create the second peak distribution
    dist2 = stats.multivariate_normal(mean=mean2, cov=cov2)

    # Create a grid of points
    x = np.linspace(-5, 8, size)
    y = np.linspace(-5, 8, size)
    X, Y = np.meshgrid(x, y)

    # Evaluate the density at each point in the grid
    Z1 = dist1.pdf(np.dstack((X, Y)))
    Z2 = dist2.pdf(np.dstack((X, Y)))

    # Combine the densities from both peaks
    pdf = Z1 / 2 + Z2
    pdf = pdf * -1 + np.max(pdf)
    return pdf, X, Y


def luetje_initial_cond():
    """
    Read in image from the Lutje paper after 5 days.
    Values are between 0-255.
    :return: 2D np array
    """
    return iio.imread('luetjesinit.png')[:,:,1]


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



if __name__ == "__main__":
    size = 100

    psi_0, X, Y = build_2d_beta(size=size)
    plot_distribution(psi_0, X, Y,"alpha=(2, 2), beta=(2, 2), size=100)")
