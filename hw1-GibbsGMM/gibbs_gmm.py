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

K = 2
dim = 2

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


X1 = np.random.multivariate_normal([5, 5], np.diag([0.5, 0.5]), size=20)
X2 = np.random.multivariate_normal([8, 8], np.diag([0.5, 0.5]), size=20)
X = np.vstack([X1, X2])

x_train_data_norm = X

N = X.shape[0]


# mu_var_dict = {}
# mu_var_dict['mu'] = np.mean(x_train_data_norm,axis=0)
# mu_var_dict['cov'] = np.cov(x_train_data_norm,rowvar=False)
# with open('./MNIST_data/mu_var_train.pickle', 'wb') as handle:
#     pickle.dump(mu_var_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


c=0.1
alpha = np.array([1]*K)
mus = []
for k in range(K):
	# sample = np.random.multivariate_normal(np.mean(x_train_data_norm,axis=0), c*np.identity(dim))
	sample = np.random.multivariate_normal(np.zeros(dim), c*np.identity(dim))
	mus.append(sample)
# sigmas = [np.cov(x_train_data_norm, rowvar=False)]*K
# sigmas = [np.identity(dim)]*10
sigmas = [np.cov(x_train_data_norm,rowvar=False)+0.1*np.identity(dim)]*K
zs = np.zeros([N]) 
theta = np.random.dirichlet(alpha)


for itr in range(50):
    probs = np.zeros([N, K])

    for k in range(K):
    	# print(np.linalg.inv(sigmas[k]))
        p = theta[k] * st.multivariate_normal.pdf(x_train_data_norm, mean=mus[k], cov=sigmas[k])
        # import pdb
        # pdb.set_trace()
        probs[:, k] = p

    # Normalize
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

        mu_head = np.matmul(np.matmul(sigmas[k],np.sum(Xk,axis=0).transpose()), np.linalg.inv(sigma_head_inv))

        mus[k] = st.multivariate_normal.rvs(mu_head, np.linalg.inv(sigma_head_inv))


    theta = np.random.dirichlet(alpha + Ns)

    print(Ns)

    # import pdb
    # pdb.set_trace()


    # for k in range(K):








