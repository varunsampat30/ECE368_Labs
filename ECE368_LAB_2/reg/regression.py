import numpy as np
import matplotlib.pyplot as plt
import util

def plot_contour(x_axis, y_axis, density, title):
    plt.figure()
    c = plt.contour(x_axis, y_axis, density)
    plt.clabel(c, inline=1, fontsize=8)
    plt.scatter([-0.1], [-0.5], label='True value of a')
    plt.legend()
    plt.title(title)
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.savefig(title)

def get_contour_x_set():
    # In all contour plots, the x-axis representsa0, 
    # and the  y-axis  representsal
    a0 = np.arange(-1, 1, 0.01)
    a1 = np.arange(-1, 1, 0.01)
    
    A0, A1 = np.meshgrid(a0, a1) 
    
    x_set = np.dstack((A0, A1))
    x_set = np.reshape(x_set, (len(A0)*len(A1), 2))
    
    return A0, A1, x_set

def gaussian_distributions(title, mu, cov_matrix):
    A0, A1, x_set = get_contour_x_set()
    density = util.density_Gaussian(
            mu, 
            cov_matrix, 
            x_set).reshape(A0.shape[0], A0.shape[1])
    plot_contour(A0, A1, density, title)
    
def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    mu = [0, 0]
    cov_matrix = [[beta, 0], [0, beta]]
    
    gaussian_distributions("prior", mu, cov_matrix)

    
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    
    ### TODO: Write your code here
    
    x_modified = []
    for i in range(len(x)):
        x_modified.append([1, x[i][0]])
    x_modified = np.array(x_modified)

    
    x_t = np.transpose(x_modified)
    
    ratio = sigma2 * ((1/beta)**2) * (np.identity(2))
    
    common_term = np.matmul(x_t, x_modified) + ratio
    common_term = np.linalg.inv(common_term)
    
    mu = np.matmul(common_term, x_t)
    mu = np.matmul(mu, z)

    mu = [mu[0][0], mu[1][0]]
    
    title = "posterior" + str(len(x))
    
    Cov = common_term*sigma2
    gaussian_distributions(title, mu, Cov)
    
    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    
    x_modified = []

    for i in range(len(x)):
        x_modified.append([1, x[i]])
        
    x_modified = np.array(x_modified)
    x_t = np.transpose(x_modified)
    
    
    z_mu = np.matmul(np.transpose(mu), x_t)
    
    """
    coefficients of the regression (w) has covariance of Sigma_{w|y}, 
    then the variance of y=wx will be x^T * Sigma_{w|y} * x
    """
    z_var = np.matmul(x_modified, np.matmul(Cov, x_t))
    
    # uncertainty is along the diagonals
    """
    uncertainty = np.zeros(len(z_var))
    for i in range(len(uncertainty)):
        uncertainty[i] = z_var[i][i]
    """
    uncertainty = z_var.diagonal()
    uncertainty.setflags(write=1) # make this writetable
    
    """
    recall z = a1*x + a0 + w
    uncertainty (the var) so far is just the variance of a1*x + a0,
    must add variance of the noise & take sqrt
    must take square root
    """
    for i in range(len(uncertainty)):
        uncertainty[i] += sigma2
    uncertainty = np.sqrt(uncertainty) # std deviation
    
    """
    plotting
    """
    plt.figure()
    plt.ylim(-4,4)    
    plt.xlim(-4,4)
    title = "predict" + str(len(x_train))
    plt.title(title)
    plt.xlabel("Input (x)")
    plt.ylabel("Prediction (z)")
    
    plt.scatter(x_train, z_train)
    plt.errorbar(x, z_mu, uncertainty)
    plt.savefig(title)
    
    return

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)

    # posterior distribution p(a|x,z)
    mu_1, Cov_1 = posteriorDistribution(x[:1],z[:1],beta,sigma2)
    mu_5, Cov_5 = posteriorDistribution(x[:5],z[:5],beta,sigma2)
    mu_all, Cov_all = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu_1,Cov_1,x[:1],z[:1])
    predictionDistribution(x_test,beta,sigma2,mu_5,Cov_5,x[:5],z[:5])
    predictionDistribution(x_test,beta,sigma2,mu_all,Cov_all,x,z)
    

   

    
    
    

    
