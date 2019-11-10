import tensorflow as tf
import numpy as np
import math
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import datetime
import math
import scipy.stats as st

import tensorflow.examples.tutorials.mnist.input_data as input_data
from sklearn.decomposition import PCA

K = 3
dim = 78

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mean_img = np.mean(mnist.train.images, axis=0)

train_total_num_sample = mnist.train._num_examples
N = train_total_num_sample
test_total_num_sample = mnist.test._num_examples

train_data = mnist.train.next_batch(train_total_num_sample)
x_train_data = train_data[0]
x_train_data_norm = np.array([img - mean_img for img in x_train_data])

test_data = mnist.test.next_batch(test_total_num_sample)
x_test_data = test_data[0]
x_test_data_norm = np.array([img - mean_img for img in x_test_data])
y_test_data = test_data[1]

x_train_data_norm = x_train_data_norm[:10000,:]


X1 = np.random.multivariate_normal([2]*dim, np.diag([0.5]*dim), size=20)
X2 = np.random.multivariate_normal([5]*dim, np.diag([0.5]*dim), size=20)
X3 = np.random.multivariate_normal([8]*dim, np.diag([0.5]*dim), size=20)
X4 = np.random.multivariate_normal([-3]*dim, np.diag([0.5]*dim), size=20)
X5 = np.random.multivariate_normal([11]*dim, np.diag([0.5]*dim), size=20)
X6 = np.random.multivariate_normal([-6]*dim, np.diag([0.5]*dim), size=20)
X7 = np.random.multivariate_normal([-11]*dim, np.diag([0.5]*dim), size=20)
X8 = np.random.multivariate_normal([-2]*dim, np.diag([0.5]*dim), size=20)
X9 = np.random.multivariate_normal([15]*dim, np.diag([0.5]*dim), size=20)
X10 = np.random.multivariate_normal([-15]*dim, np.diag([0.5]*dim), size=20)

X = np.vstack([X1, X2])
X = np.vstack([X1, X2, X3, X4, X5, X6, X7, X8, X9, X10])
X = np.vstack([X1, X2, X3])

x_train_data_norm = X

# print(x_train_data_norm.shape)
# import sys
# sys.exit()

N = x_train_data_norm.shape[0]

# pca = PCA(n_components=196)
# x_train_data_norm = pca.fit_transform(x_train_data_norm)


# mu_var_dict = {}
# mu_var_dict['mu'] = np.mean(x_train_data_norm,axis=0)
# mu_var_dict['cov'] = np.cov(x_train_data_norm,rowvar=False)
# with open('./MNIST_data/mu_var_train.pickle', 'wb') as handle:
#     pickle.dump(mu_var_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


c=0.1
alpha = np.array([1]*K)
mus = []
m0 = [np.array([1]*dim)]*K
for k in range(K):
	sample = np.random.multivariate_normal(np.mean(x_train_data_norm,axis=0), c*np.identity(dim))
	sample = np.random.multivariate_normal(np.zeros((dim)), c*np.identity(dim))
	# sample = np.random.multivariate_normal(np.zeros(dim), c*np.identity(dim))
	# print(sample.shape)
	# import sys
	# sys.exit()
	mus.append(sample)
	# m0.append(sample)
mus = np.array([[1, 1], [15, 15], [35,35]], dtype='float')
# m0 = np.array([[1, 1], [1, 1]], dtype='float')
# sigmas = [np.cov(x_train_data_norm, rowvar=False)]*K
# sigmas = [np.identity(dim)]*10
# mus = np.array([[1, 1], [15, 15]], dtype='float')
# mus = m0
# m0 = np.array([[1, 1], [1, 1]], dtype='float')
sigmas = [np.diag([1]*dim)]*K
# sigmas = np.array([np.diag([1, 1]), np.diag([1, 1])], dtype='float')
zs = np.zeros([N]) 
theta = np.random.dirichlet(alpha)


for itr in range(50):
    probs = np.zeros([N, K])

    for k in range(K):
    	# print(np.linalg.inv(sigmas[k]))
        # p = math.log(theta[k]) - (dim/2.0)*math.log(2*math.pi)- 0.5*math.log(np.linalg.det(sigmas[k]))
        # - 0.5*np.matmul(np.matmul((x_train_data_norm - mus[k]), np.linalg.inv(sigmas[k])), (x_train_data_norm - mus[k]).T)
        # p1 = theta[k] * st.multivariate_normal.pdf(x_train_data_norm, mean=mus[k], cov=sigmas[k])
        # import pdb
        # pdb.set_trace()
        p = math.log(theta[k]) \
            - 0.5*np.sum(np.multiply(np.matmul((x_train_data_norm - mus[k]), np.linalg.inv(sigmas[k])), (x_train_data_norm - mus[k])), axis=1)
        p = np.exp(p)
    	# import pdb
    	# pdb.set_trace()
        probs[:, k] = p

    # Normalize
    # import pdb
    # pdb.set_trace()
    probs /= np.sum(probs, axis=1)[:, np.newaxis]
    # import pdb
    # pdb.set_trace()

    for i in range(N):
        z = np.random.multinomial(n=1, pvals=probs[i]).argmax()
        zs[i] = z


    Ns = np.zeros(K, dtype='int')
    for k in range(K):
        Xk = x_train_data_norm[zs == k]
        Ns[k] = Xk.shape[0]
        sigma_head_inv = Ns[k]*np.linalg.inv(sigmas[k]) + np.linalg.inv(c*np.identity(dim))

        left = np.linalg.inv(sigma_head_inv)
        right = np.matmul(sigmas[k],np.sum(Xk,axis=0).transpose()) + np.matmul(np.linalg.inv(c*np.identity(dim)),m0[k])
        mu_head = np.matmul(left,right)

        mus[k] = st.multivariate_normal.rvs(mu_head, np.linalg.inv(sigma_head_inv))


    theta = np.random.dirichlet(alpha + Ns)

    print(Ns)


    obj = 0
    for i in range(N):
    	obj += math.log(st.multivariate_normal.pdf(x_train_data_norm[i], mean=mus[int(zs[i])], cov=sigmas[int(zs[i])]))
    print(obj)


    # import pdb
    # pdb.set_trace()


    # for k in range(K):
    # 	print('{} data in cluster-{}, mean: {}'.format(Ns[k], k, mus[k]))
    for k in range(K):
    	print('{} data in cluster-{}'.format(Ns[k], k))







"""
Posterior sampling for Gaussian Mixture Model using Gibbs sampler
"""
# import numpy as np
# import scipy.stats as st
# import matplotlib.pyplot as plt
# import math


# K = 2
# dim = 2

# # Generate data
# X1 = np.random.multivariate_normal([5, 5], np.diag([0.5, 0.5]), size=20)
# X2 = np.random.multivariate_normal([8, 8], np.diag([0.5, 0.5]), size=20)
# X = np.vstack([X1, X2])

# N = X.shape[0]

# # GMM params
# mus = np.array([[1, 1], [15, 15]], dtype='float')
# sigmas = np.array([np.diag([1, 1]), np.diag([1, 1])], dtype='float')
# lambdas = np.array([np.linalg.inv(sigmas[0]), np.linalg.inv(sigmas[1])])
# pis = np.array([0.5, 0.5])  # Mixing probs.
# zs = np.zeros([N])  # Assignments

# # Priors
# alpha = np.array([1, 1])
# pis = np.random.dirichlet(alpha)
# mus0 = np.array([[1, 1], [1, 1]], dtype='float')
# sigmas0 = np.array([np.diag([1, 1]), np.diag([1, 1])], dtype='float')
# lambdas0 = np.array([np.linalg.inv(sigmas0[0]), np.linalg.inv(sigmas0[1])])

# # Gibbs sampler
# for it in range(50):
#     # Sample from full conditional of assignment
#     # z ~ p(z) \propto pi*N(y|pi)
#     probs = np.zeros([N, K])

#     for k in range(K):
#         p = pis[k] * st.multivariate_normal.pdf(X, mean=mus[k], cov=sigmas[k])
#         p1 = math.log(pis[k]) - (dim/2.0)*math.log(2*math.pi)- 0.5*math.log(np.linalg.det(sigmas[k]))\
#             - 0.5*np.sum(np.multiply(np.matmul((X - mus[k]), np.linalg.inv(sigmas[k])), (X - mus[k])), axis=1)
#         p1 = np.exp(p1)
#         # import pdb
#         # pdb.set_trace()
#         probs[:, k] = p1

#     # Normalize
#     probs /= np.sum(probs, axis=1)[:, np.newaxis]

#     # For each data point, draw the cluster assignment
#     for i in range(N):
#         z = np.random.multinomial(n=1, pvals=probs[i]).argmax()
#         zs[i] = z

#     # Sample from full conditional of cluster parameter
#     # Assume fixed covariance => posterior is Normal
#     # mu ~ N(mu, sigma)
#     Ns = np.zeros(K, dtype='int')

#     for k in range(K):
#         # Gather all data points assigned to cluster k
#         Xk = X[zs == k]
#         Ns[k] = Xk.shape[0]

#         # Covariance of posterior
#         lambda_post = lambdas0[k] + Ns[k]*lambdas[k]
#         cov_post = np.linalg.inv(lambda_post)

#         # Mean of posterior
#         left = cov_post
#         right = np.matmul(lambdas0[k],mus0[k]) + np.matmul(Ns[k]*lambdas[k],np.mean(Xk, axis=0))
#         mus_post = np.matmul(right,left)

#         # Draw new mean sample from posterior
#         mus[k] = st.multivariate_normal.rvs(mus_post, cov_post)

#     # Sample from full conditional of the mixing weight
#     # pi ~ Dir(alpha + n)
#     pis = np.random.dirichlet(alpha + Ns)
#     print(Ns)

# # Expected output:
# # ----------------
# # 20 data in cluster-0, mean: [ 5  5 ]
# # 20 data in cluster-1, mean: [ 8  8 ]
# for k in range(K):
#     print('{} data in cluster-{}, mean: {}'.format(Ns[k], k, mus[k]))