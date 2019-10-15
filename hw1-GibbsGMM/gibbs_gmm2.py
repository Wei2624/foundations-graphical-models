# import tensorflow as tf
# import numpy as np
# import math
# import pickle
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import datetime
# import math
# import scipy.stats as st

# import tensorflow.examples.tutorials.mnist.input_data as input_data
# from sklearn.decomposition import PCA

# K = 3
# dim = 78


# X1 = np.random.multivariate_normal([2]*dim, np.diag([0.5]*dim), size=20)
# X2 = np.random.multivariate_normal([5]*dim, np.diag([0.5]*dim), size=20)
# X3 = np.random.multivariate_normal([8]*dim, np.diag([0.5]*dim), size=20)

# X = np.vstack([X1, X2, X3])


# x_train_data_norm = X

# N = x_train_data_norm.shape[0]

# c=1
# alpha = np.array([1]*K)
# mus = []
# m0 = [np.array([1]*dim)]*K

# for k in range(K):
# 	# sample = np.random.multivariate_normal(np.mean(x_train_data_norm,axis=0), c*np.identity(dim))
# 	sample = np.random.multivariate_normal(np.zeros((dim)), c*np.identity(dim))
# 	mus.append(sample)

# sigmas = [np.diag([1]*dim)]*K
# theta = np.random.dirichlet(alpha)
# zs = np.zeros([N]) 

# probs = np.zeros([N, K])

# for k in range(K):
#     p = math.log(theta[k]) \
#         - 0.5*np.sum(np.multiply(np.matmul((x_train_data_norm - mus[k]), np.linalg.inv(sigmas[k])), (x_train_data_norm - mus[k])), axis=1)
#     p = np.exp(p)
#     probs[:, k] = p

# # Normalize
# probs /= np.sum(probs, axis=1)[:, np.newaxis]


# for i in range(N):
#     z = np.random.multinomial(n=1, pvals=probs[i]).argmax()
#     zs[i] = z


# obj_itr = np.zeros((50))


# for itr in range(50):
#     probs = np.zeros([N, K])

#     for k in range(K):
#         p = math.log(theta[k]) \
#             - 0.5*np.sum(np.multiply(np.matmul((x_train_data_norm - mus[k]), np.linalg.inv(sigmas[k])), (x_train_data_norm - mus[k])), axis=1)
#         p = np.exp(p)
#         probs[:, k] = p

#     probs /= np.sum(probs, axis=1)[:, np.newaxis]
#     for i in range(N):
#         z = np.random.multinomial(n=1, pvals=probs[i]).argmax()
#         zs[i] = z


#     Ns = np.zeros(K, dtype='int')
#     for k in range(K):
#         Xk = x_train_data_norm[zs == k]
#         Ns[k] = Xk.shape[0]
#         sigma_head_inv = Ns[k]*np.linalg.inv(sigmas[k]) + np.linalg.inv(c*np.identity(dim))

#         left = np.linalg.inv(sigma_head_inv)
#         right = np.matmul(sigmas[k],np.sum(Xk,axis=0).transpose()) + np.matmul(np.linalg.inv(c*np.identity(dim)),m0[k])
#         mu_head = np.matmul(left,right)

#         mus[k] = st.multivariate_normal.rvs(mu_head, np.linalg.inv(sigma_head_inv))


#     theta = np.random.dirichlet(alpha + Ns)

#     print(Ns)

#     obj = 0
#     for i in range(N):
#     	obj += math.log(st.multivariate_normal.pdf(x_train_data_norm[i], mean=mus[int(zs[i])], cov=sigmas[int(zs[i])]))
#     obj_itr[itr] = obj
#     print(obj)


#     # import pdb
#     # pdb.set_trace()


#     for k in range(K):
#     	print('{} data in cluster-{}, mean: {}'.format(Ns[k], k, mus[k]))
#     # for k in range(K):
#     # 	print('{} data in cluster-{}'.format(Ns[k], k))

# plt.plot(range(50),obj_itr)
# plt.xlabel('iterations')
# plt.ylabel('objective')
# plt.show()



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

K = 10
dim = 196

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mean_img = np.mean(mnist.train.images, axis=0)

train_total_num_sample = mnist.train._num_examples
N = train_total_num_sample
test_total_num_sample = mnist.test._num_examples

train_data = mnist.train.next_batch(train_total_num_sample)
x_train_data = train_data[0]
x_train_data_norm = np.array([img - mean_img for img in x_train_data])
y_train_data = train_data[1]

pca = PCA(n_components=196)
x_train_data_norm = pca.fit_transform(x_train_data_norm)

x_norm_subsample = np.zeros((1000,196))
for k in range(K):
	idx = np.where(y_train_data[:,k] == 1)
	x_norm_subsample[k*100:(k+1)*100,:] = x_train_data_norm[idx[0][:100],:]

N = x_norm_subsample.shape[0]



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
	# sample = np.random.multivariate_normal(np.mean(x_train_data_norm,axis=0), c*np.identity(dim))
	sample = np.random.multivariate_normal(np.zeros((dim)), c*np.identity(dim))
	# sample = np.random.multivariate_normal(np.zeros(dim), c*np.identity(dim))
	# print(sample.shape)
	# import sys
	# sys.exit()
	mus.append(sample)
	# m0.append(sample)
# mus = np.array([[1, 1], [15, 15], [35,35]], dtype='float')
# m0 = np.array([[1, 1], [1, 1]], dtype='float')
# sigmas = [np.cov(x_train_data_norm, rowvar=False)]*K
# sigmas = [np.identity(dim)]*10
# mus = np.array([[1, 1], [15, 15]], dtype='float')
# mus = m0
# m0 = np.array([[1, 1], [1, 1]], dtype='float')
sigmas = [np.diag([0.1]*dim)]*K
# sigmas = np.array([np.diag([1, 1]), np.diag([1, 1])], dtype='float')
theta = np.random.dirichlet(alpha)

zs = np.zeros([N]) 

probs = np.zeros([N, K])

for k in range(K):
    p = math.log(theta[k]) \
        - 0.5*np.sum(np.multiply(np.matmul((x_norm_subsample - mus[k]), np.linalg.inv(sigmas[k])), (x_norm_subsample - mus[k])), axis=1)
    p = np.exp(p)
    probs[:, k] = p

# Normalize
probs /= np.sum(probs, axis=1)[:, np.newaxis]


for i in range(N):
    z = np.random.multinomial(n=1, pvals=probs[i]).argmax()
    zs[i] = z


for itr in range(500):
    probs = np.zeros([N, K])

    for k in range(K):
    	# print(np.linalg.inv(sigmas[k]))
        # p = math.log(theta[k]) - (dim/2.0)*math.log(2*math.pi)- 0.5*math.log(np.linalg.det(sigmas[k]))
        # - 0.5*np.matmul(np.matmul((x_train_data_norm - mus[k]), np.linalg.inv(sigmas[k])), (x_train_data_norm - mus[k]).T)
        # p1 = theta[k] * st.multivariate_normal.pdf(x_train_data_norm, mean=mus[k], cov=sigmas[k])
        # import pdb
        # pdb.set_trace()
        p = math.log(theta[k]) \
            - 0.5*np.sum(np.multiply(np.matmul((x_norm_subsample - mus[k]), np.linalg.inv(sigmas[k])), (x_norm_subsample - mus[k])), axis=1)
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
        Xk = x_norm_subsample[zs == k]
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
    	# obj += math.log(st.multivariate_normal.pdf(x_norm_subsample[i], mean=mus[int(zs[i])], cov=sigmas[int(zs[i])]))
    	# import pdb
    	# pdb.set_trace()
    	obj += 0.5*np.sum(np.multiply(np.matmul((x_norm_subsample[i] - mus[int(zs[i])]), np.linalg.inv(sigmas[int(zs[i])])), (x_norm_subsample[i] - mus[int(zs[i])])))
    print(obj)


    # import pdb
    # pdb.set_trace()


    # for k in range(K):
    # 	print('{} data in cluster-{}, mean: {}'.format(Ns[k], k, mus[k]))
    for k in range(K):
    	print('{} data in cluster-{}'.format(Ns[k], k))
