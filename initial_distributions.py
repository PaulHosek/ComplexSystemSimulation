import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import imageio.v3 as iio
from scipy import ndimage


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

def order_distribution(control_parameter, size):
    """
    Generate distribution with some order value.
    Log(control_parameter) maps linearly to the entropy-based order parameter.
    Output distribution will consist of uniformly distributed numbers in the half-open interval [0.0, 1.0).
    :param control_parameter (float): In [0,1]. Determins the order of the topology.
    :param size (int): Sidelength of the topology.
    :return (2d np.array): Topology; z-lattice only.
    """

    random_distribution = np.random.random((size, size))
    uniform = np.zeros((size, size))
    num_elements = int(size * size * control_parameter)
    indices = np.random.choice(size * size, size=num_elements, replace=False)
    uniform.flat[indices] = random_distribution.flat[indices]
    return uniform


def calculate_order_parameter(distribution=None, control_parameter=None, size=100):
    """
    Calculate the order parameter for a given control parameter or 2d Distribution.
    Order parameter is transformed mean entropy of the system.
    Scales linearly with control parameter. Is min-max scaled.
    :param control_parameter (float): The control parameter that influences the level of order.
    :return: Order parameter value.
    """
    if distribution is None and control_parameter is None:
        print(control_parameter)
        raise Exception("Provide either an input distribution or a control parameter.")
    if distribution:
        assert len(distribution.shape) == 2, "Input distribution is not 2D."
        size = len(distribution)
    if control_parameter:
        random_distribution = np.random.random((size, size))
        distribution = np.zeros((size, size))
        num_elements = int(size * size * control_parameter)
        indices = np.random.choice(size * size, size=num_elements, replace=False)
        distribution.flat[indices] = random_distribution.flat[indices]

    order_parameter = (np.nanmean(stats.entropy(distribution)))
    min_order = 2.05625
    max_order = 4.41097
    return (order_parameter - min_order) / (max_order - min_order) * (np.log(100)/np.log(size))

def plot_control_order_curve():
    """
    Plot the relationship of order and control parameter.
    :return: None
    """
    control_parameter_range = np.linspace(0.1, 1, 20)
    order_parameters = []

    for control_parameter in control_parameter_range:
        order_parameter = calculate_order_parameter(control_parameter=control_parameter)
        order_parameters.append(order_parameter)

    plt.plot(control_parameter_range, order_parameters)
    plt.xlabel("Control Parameter")
    plt.ylabel("Order Parameter")
    plt.title("Order Parameter vs Control Parameter")
    plt.show()


if __name__ == "__main__":
    size = 100

    psi_0, X, Y = build_2d_beta(size=size)
    plot_distribution(psi_0, X, Y,"alpha=(2, 2), beta=(2, 2), size=100)")

# code from Popovic et al., 2020 (https://doi.org/10.1029/2019JC016029)
# available under https://zenodo.org/record/3930528

"""
A function to create a synthetic topography. Returns a numpy array of size (res,res).
    res - size of the output array
    mode - topography type. Options - 'snow_dune', 'diffusion', and 'rayleigh'
    tmax - time to diffuse a random configuration in 'diffusion' and 'rayleigh' topographies. Controls the typical length-scale
    dt - time-step for diffusion in 'diffusion' and 'rayleigh' topographies. If too large, creating a topography may fail
    g - anisotropy parameter
    sigma_h - standard deviation of the topography
    h - mean elevation
    snow_dune_radius - mean radius of mounds in the 'snow_dune' topography. Controls the typical length-scale
    Gaussians_per_pixel - density of mounds in the 'snow_dune' topography (number of mounds * snow_dune_radius^2 / res^2)
    number_of_r_bins - number of categories of mound radii to consider in the 'snow_dune' topography
    window_size - cutoff parameter for placing mounds in the 'snow_dune' topography
    snow_dune_height_exponent - exponent that relates mound radius and mound height in the 'snow_dune' topography 
"""

def Create_Initial_Topography(res = 500, mode = 'snow_dune',tmax = 2,dt = 0.1, g = 1,sigma_h = 1., h = 0., snow_dune_radius = 1., Gaussians_per_pixel = 0.2, 
                              number_of_r_bins = 150, window_size = 5, snow_dune_height_exponent = 1.):
                              

    if mode == 'diffusion':
        t = np.arange(0,tmax,dt)
        ice_topo = 0.5-np.random.rand(res,res)
        stencil = np.array([[0, g, 0],[1, -2*(1+g), 1], [0, g, 0]])
        
        for i in range(1,len(t)):
            ice_topo += dt*ndimage.convolve(ice_topo, stencil)
            
    if mode == 'rayleigh':
        t = np.arange(0,tmax,dt)
        ice_topo1 = 0.5-np.random.rand(res,res)
        ice_topo2 = 0.5-np.random.rand(res,res)
        stencil = np.array([[0, g, 0],[1, -2*(1+g), 1], [0, g, 0]])
        
        for i in range(1,len(t)):
            ice_topo1 += dt*ndimage.convolve(ice_topo1, stencil)
            ice_topo2 += dt*ndimage.convolve(ice_topo2, stencil)
        
        ice_topo = np.sqrt(ice_topo1**2+ice_topo2**2)
        
    if mode == 'snow_dune':
        ice_topo = np.zeros([res,res])
        N = np.ceil((res/snow_dune_radius)**2 * Gaussians_per_pixel).astype(int)
        r0 = np.random.exponential(snow_dune_radius,N)
        
        bins = np.linspace(np.min(r0),np.max(r0),number_of_r_bins+1)
        r0_bins = np.zeros(number_of_r_bins)
        r0_N = np.zeros(number_of_r_bins).astype(int)
        
        for i in range(1,number_of_r_bins):
            loc = (r0 >= bins[i-1]) & (r0 < bins[i])
            r0_bins[i] = np.mean(r0[loc])
            r0_N[i] = np.sum(loc)

        r0_bins = r0_bins[r0_N>0] 
        r0_N = r0_N[r0_N>0] 
        
        for i in range(len(r0_bins)):
            r = r0_bins[i]
            h0 = r**snow_dune_height_exponent / snow_dune_radius**snow_dune_height_exponent
            cov = np.eye(2); cov[1,1] = g
            cov *= r**2
            
            rv = stats.multivariate_normal([0,0], cov) 
            
            x0 = np.random.choice(np.arange(-res/2,res/2),r0_N[i])
            y0 = np.random.choice(np.arange(-res/2,res/2),r0_N[i])
        
            x = np.arange(-np.ceil(r*window_size).astype(int),np.ceil(r*window_size).astype(int)+1)
            y = x.copy()
            X,Y = np.meshgrid(x,y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X; pos[:, :, 1] = Y
            
            G = rv.pdf(pos) * 2 * np.pi * np.sqrt(np.linalg.det(cov)) * h0
            
            for j in range(r0_N[i]):
                loc_x = ((pos[:,:,0]+x0[j]) % res).astype(int)
                loc_y = ((pos[:,:,1]+y0[j]) % res).astype(int)
        
                ice_topo[loc_x,loc_y] += G
        
    ice_topo /= np.std(ice_topo)
    ice_topo -= np.mean(ice_topo)
    ice_topo *= sigma_h
    ice_topo += h
    return ice_topo      

