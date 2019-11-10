import sys
import os
import numpy as np
from scipy import special
from math import exp


digamma = special.digamma


doc_count_path = '../../HW2/hw2-datasets/ap/ap.dat'
voc_index_path = '../../HW2/hw2-datasets/ap/vocab.txt'
voc_list = []
word_count_list = []
word_index_dict_list = []

#######################################################

held_out_prop = 0.2
total_doc = 2246
total_words = 10473
total_topics = 50
total_train_index = int(total_doc*(1 - held_out_prop))

#######################################################

with open(voc_index_path) as f:
	for line in f:
		voc_list.append(line.strip())

V_dim = len(voc_list)

with open(doc_count_path) as f:
	for line in f:
		parts = line.strip().split(' ')
		pair_dict = {}
		# word_count_list.append(int(parts[0]))
		del parts[0]
		for pair in parts:
			each = pair.split(':')
			pair_dict[each[0]] = int(each[1])
		word_index_dict_list.append(pair_dict)
		word_count_list.append(int(sum(word_index_dict_list[-1].values())))

in_samples_word_count = word_count_list[:total_train_index]
in_samples_word_index_dict = word_index_dict_list[:total_train_index]
out_sample_word_count = word_index_dict_list[total_train_index:]
out_sample_word_index_dict = word_index_dict_list[total_train_index:]

#####################################################

lambdas = np.random.rand(total_topics,total_words)
gammas = np.ones((total_doc,total_topics))
phis = []
z = []

for count in word_count_list:
	phi_jk = np.zeros((count,total_topics))
	phi_jk[:,:] = 1.0/total_topics
	phis.append(phi_jk)
	z.append(np.zeros((count)))

#####################################################

iterations = 50
ita = np.ones((total_topics,total_words))
alpha = np.array([1.0]*total_topics)


#####################################################

for itr in range(iterations):
	print '-----------------------update phis---------------------------'
	for i,word_count in enumerate(in_samples_word_count):
		accumulate_idx = 0
		theta_digamma = digamma(gammas[i,:]) - digamma(np.sum(gammas[i,:]))
		print 'document for phis: ',i 
		for key, val in in_samples_word_index_dict[i].iteritems():
			for k in range(total_topics):
				beta_digamma = digamma(lambdas[k,int(key)]) - digamma(np.sum(lambdas[k,:]))
				phis[i][accumulate_idx:accumulate_idx+val,k] = exp(beta_digamma + theta_digamma[k])
			accumulate_idx += val
			# if accumulate_idx >= word_count:
			# 	import pdb
			# 	pdb.set_trace()
		phis[i][:,:] /= np.sum(phis[i][:,:], axis=1)[:, np.newaxis]

	# for i,word_count in enumerate(in_samples_word_count):
	# 	for j in range(word_count):
	# 		topic = np.random.multinomial(n=1, pvals=phis[i][j,:]).argmax()
	# 		z[i][j] = topic

	print '---------------------update gammas-------------------------'
	for i,word_count in enumerate(in_samples_word_count):
		print 'document for gammas:',i
		for k in range(total_topics):
			probs = 0
			for j in range(word_count):
				probs +=  phis[i][j,k]
			gammas[i,k] = alpha[k] + probs

	print '-----------------------update lambdas-----------------------'
	for k in range(total_topics):
		for v in range(total_words):
			print 'topic for lambdas', k, 'word: ', v
			counts = 0
			for i,word_count in enumerate(in_samples_word_count):
				 j = 0
				 for key, val in in_samples_word_index_dict[i].iteritems():
				 	if int(key) == v:
				 		for l in range(val):
				 			if j+l >= phis[i][:,:].shape[0]:
				 				import pdb
				 				pdb.set_trace()
				 			counts += phis[i][j+l,k]
				 		break
				 	else:
				 		j += val
			lambdas[k,v] = ita[k,v] + counts

	print "iteration: ", itr





