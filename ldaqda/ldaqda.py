import numpy as np
import matplotlib.pyplot as plt
import util

def meanEstimator(x):
    """
    - Compute the mean (MLE) of a normal variable
    """
    height = x[:, 0]
    weight = x[:, 1]
    
    return np.sum(height)/len(height), np.sum(weight)/len(weight) 

def splitData(x, y):
    male_x = x[y == 1]
    female_x = x[y == 2]
    return male_x, female_x
    
def covariance(x, y, mu_x, mu_y):
    result = np.sum((x-mu_x)*(y-mu_y))/(len(x))
    return result

def covarianceMatrix(x, mean_vector):
    diff_x_mu = np.zeros((len(x), 2))
    
    for j in range(len(x)):
        diff_x_mu[j] = x[j]-mean_vector
    x_transpose = diff_x_mu.transpose()
    cov_matrix = (1/len(x))*(np.matmul(x_transpose, diff_x_mu))
    return cov_matrix

def LDA_formula(covMatrix, male_mean, female_mean, X, Y):
    invMatrix = np.linalg.inv(covMatrix)
    A = np.transpose(np.matmul(invMatrix, male_mean)) - np.transpose(np.matmul(invMatrix, female_mean))
    
    b = 0.5*(np.matmul(np.transpose(male_mean), np.matmul(invMatrix, male_mean)) - 
             np.matmul(np.transpose(female_mean), np.matmul(invMatrix, female_mean)))
    return A[0]*X + A[1]*Y - b

def LDA(prior, covMatrix, male_mean, female_mean):
    # A * xT = b

    x = np.arange(50, 80, 0.15)
    y = np.arange(80, 280, 1)
    X, Y = np.meshgrid(x, y)
    
    Z = LDA_formula(covMatrix, male_mean, female_mean, X, Y)
    return X, Y, Z

def QDA_formula(cov_male, cov_female, mu_male, mu_female, X, Y):
    inv_male = np.linalg.inv(cov_male)
    inv_female = np.linalg.inv(cov_female)
    
    det_male = np.linalg.det(cov_male)
    det_female = np.linalg.det(cov_female)
    
    A = np.transpose(np.matmul(inv_male, mu_male)) - np.transpose(np.matmul(inv_female, mu_female))
    b = 0.5*(np.matmul(np.transpose(mu_male), np.matmul(inv_male, mu_male)) - np.matmul(np.transpose(mu_female), np.matmul(inv_female, mu_female)))
    return 0.5*np.log(det_male) - 0.5*np.log(det_female) + b - A[0]*X - A[1]*Y\
       + 0.5*X*(inv_male[0, 0]*X+inv_male[0, 1]*Y) + 0.5*Y*(inv_male[1, 0]*X\
       +inv_male[1, 1]*Y) - 0.5*X*(inv_female[0, 0]*X+inv_female[0, 1]*Y)\
       - 0.5*Y*(inv_female[1, 0]*X+inv_female[1,1]*Y) 

def QDA(prior, cov_male, cov_female, mu_male, mu_female):
    
    x = np.arange(50, 80, 0.15)
    y = np.arange(80, 280, 1)
    X, Y = np.meshgrid(x, y) 
    
    Z = QDA_formula(cov_male, cov_female, mu_male, mu_female, X, Y)
    return X, Y, Z


def dataScatterPlot(male_x, female_x):    
    plt.figure()
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title("Training data with the LDA boundary curve")
    plt.scatter(male_x[:, 0], male_x[:, 1], color='blue', label='Males')
    plt.scatter(female_x[:, 0], female_x[:, 1], color='red', label='Females')
    
def plotDecisionBoundary(height, weight, result):
    plt.contour(height, weight, result, 0)

def gaussianContour(height, weight, density_male, density_female):
    plt.contour(height, weight, density_male)
    plt.contour(height, weight, density_female)
    plt.legend(loc='upper left')
    
def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    male_x, female_x = splitData(x, y)
    
    ## mean
    mu_male = meanEstimator(male_x)
    mu_female = meanEstimator(female_x)
    mu = meanEstimator(x)
    
    ## mean values print
    #print("male mu: ", mu_male)
    #print("female mu: ", mu_female)
    
    ## covariance values
    cov = covarianceMatrix(x, mu)
    cov_male = covarianceMatrix(male_x, mu_male)
    cov_female = covarianceMatrix(female_x, mu_female)

    #print("Overall cov. matrix:\n", cov)    
    #print("Male cov. matrix:\n", cov_male)
    #print("Female cov. matrix:\n", cov_female)
    
    # LDA calc.
    height, weight, result = LDA([0.5, 0.5], cov, mu_male, mu_female)
    
    #QDA calc.
    height_Q, weight_Q, result_Q = QDA([0.5, 0.5], 
                                       cov_male, 
                                       cov_female, 
                                       mu_male, 
                                       mu_female)
    
    ## gaussian calculations
    x_set = np.dstack((height, weight))
    x_set = np.reshape(x_set, (len(height)*len(weight), 2))
    density_male = util.density_Gaussian(mu_male, cov_male, x_set).reshape(height.shape[0], height.shape[1])
    density_female = util.density_Gaussian(mu_female, cov_female, x_set).reshape(height.shape[0], height.shape[1])
    
    ## figure 1 (just scatter)
    dataScatterPlot(male_x, female_x)
    plotDecisionBoundary(height, weight, result)
    gaussianContour(height, weight, density_male, density_female)
    plt.savefig("lda.pdf")
    
    #figure 2 (QDA)
    dataScatterPlot(male_x, female_x)
    plotDecisionBoundary(height, weight, result_Q)
    gaussianContour(height, weight, density_male, density_female)
    plt.savefig("qda.pdf")
    
    return (mu_male,mu_female,cov,cov_male,cov_female)

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    
    male_x, female_x = splitData(x, y)
        

    misHit = 0
    for i in range(len(x)):
        if(y[i] == 1): #supposed to be male
            if(LDA_formula(cov, mu_male, mu_female, x[i, 0], x[i, 1]) < 0):
                misHit += 1
        elif(y[i] == 2): #supposed to be a female
            if(LDA_formula(cov, mu_male, mu_female, x[i, 0], x[i, 1]) > 0):
                misHit += 1    
    
    misHit_Q = 0
    for i in range(len(x)):
        if(y[i] == 1): #supposed to be male
            if(QDA_formula(cov_male, cov_female, mu_male, mu_female, x[i, 0], x[i, 1]) > 0):
                misHit_Q += 1
        elif(y[i] == 2): #supposed to be a female
            if(QDA_formula(cov_male, cov_female, mu_male, mu_female, x[i, 0], x[i, 1]) < 0):
                misHit_Q += 1
    mis_lda, mis_qda = misHit/len(x), misHit_Q/len(x)
    print("LDA error rate: ", mis_lda)
    print("QDA error rate: ", mis_qda)

    
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    

    
    
    

    
