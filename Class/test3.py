import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

np.random.seed(123)


n = 1000
d = 2
m = 100
reg_coef = 0.1
b = 2
n_centroids = 100


# Explanatory variables are uniformly distributed
X = np.random.uniform(-4, 4, size=(n, d))
# Response is a multivariate normal
target_normal = multivariate_normal(mean=np.random.normal(size=d), cov=np.eye(d))
y = target_normal.pdf(X)

Xhat = np.random.uniform(-4, 4, size=(n, d))
yactual = target_normal.pdf(Xhat)

def compute_sigmasq(X):
    xcross = np.dot(X, X.T)
    xnorms = np.repeat(np.diag(np.dot(X, X.T)).reshape(1, -1), np.size(X, axis=0), axis=0)
    return(np.median(xnorms - 2*xcross + xnorms.T))
# Find sigmasquared for the rbf
sigmasq = compute_sigmasq(X)

def get_centroids(X, n_centroids):
    # Find the indeces
    idx = np.random.randint(np.size(X, axis=0), size=n_centroids)
    # Use indeces to grab rows
    return(X[idx, :])

centroids = get_centroids(X, n_centroids)


## After this line my implementations #####


def cal_distance(x,cent_i):
    ## calculate the distance between the centroid and the other elements
    return(np.sum((x - cent_i)**2))


phi_train = np.ones((n,1))
for i in range(n_centroids):
    cent_i  = centroids[i,:] # get the ith centroid
    dist_i = np.apply_along_axis(cal_distance , 1, X,cent_i) ## distance matrix for i th centroid
    phi_i = np.exp(-dist_i/sigmasq) ##
    phi_i = np.reshape(phi_i,(n,1)) # dummy rehsape to (n,1)
    phi_train = np.hstack((phi_train,phi_i)) # horizontally stack the matrix

inv_phi = np.linalg.pinv(phi_train) # get the psudoinverse for regular linear regression

#linear regression coefficiet
coefs = np.matmul(inv_phi,y) ## compute the weights
coefs = np.reshape(coefs,(n_centroids+1,1))  ## dummy reshape

## create phi test
phi_test = np.ones((n,1))
for i in range(n_centroids):
    cent_i  = centroids[i,:] # get the ith centroid
    dist_i = np.apply_along_axis(cal_distance , 1, Xhat,cent_i) ## distance matrix for i th centroid
    phi_i = np.exp(-dist_i/sigmasq)
    phi_i = np.reshape(phi_i,(n,1)) # dummy rehsape
    phi_test = np.hstack((phi_test,phi_i))


y_predict = np.matmul(phi_test,coefs)

plt.plot(yactual,y_predict,'*')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('n_centroid = ' +  str(n_centroids))
plt.show()
