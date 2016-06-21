# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME="indicnlp/indic_nlp_library"
# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES="indicnlp/indic_nlp_resources"

import sys
sys.path.append('{}/src'.format(INDIC_NLP_LIB_HOME))

from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp import loader
loader.load()

from indicnlp.script import  indic_scripts as isc

import tensorflow as tf

# from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from collections import defaultdict

def get_feature_vector(char_, lang):
	if(char_ == 'GO'):
		a=np.zeros([41])
		a[38]=0
		return a
	elif(char_ == 'EOW'):
		a=np.zeros([41])
		a[39]=0
		return a
	elif(char_ == 'PAD'):
		a=np.zeros([41])
		a[40]=0
		return a
	else:
		return np.append(isc.get_phonetic_feature_vector(char_.decode('utf-8'),lang),[0.,0.,0.])

def get_batch(batch_size, sequences, masks, lengths):
	rp = (np.random.permutation(len(sequences)))[:batch_size]
	return np.asarray([sequences[i] for i in rp]),np.asarray([masks[i] for i in rp]),np.asarray([lengths[i] for i in rp])

# Compute Hidden Representation
# Output and states of encoder
def compute_hidden_representation(sequences,sequence_lengths,cell,embed_W):
	x = tf.transpose(tf.add(tf.nn.embedding_lookup(embed_W,sequences),embed_b),[1,0,2])
	x = tf.reshape(x,[-1,embedding_size])
	x = tf.split(0,max_sequence_length,x)
	outputs, states = rnn.rnn(cell,x, dtype = tf.float32, sequence_length=sequence_lengths)
	
	return states

# Loss function
def seq_loss(target_sequence, target_masks, cell, out_W, out_b, initial_state, max_sequence_length, vocab_size):
	state = initial_state
	loss = 0.0
	for i in range(max_sequence_length):
		if(i==0):
			current_emb = tf.tile(first_dec_input,[batch_size,1])
			# current_emb = tf.zeros_like(tf.nn.embedding_lookup(embed_W,target_sequence[:,0]))
		else:
			current_emb = tf.nn.embedding_lookup(embed_W,target_sequence[:,i-1])+embed_b
			# current_emb = tf.nn.embedding_lookup(embed_W,output_id)+embed_b
		# print current_emb
		if i > 0 : tf.get_variable_scope().reuse_variables()
		# TODO Porpose of above line
		output,state = cell(current_emb,state)

		labels = tf.expand_dims(target_sequence[:,i],1)
		indices = tf.expand_dims(tf.range(0,batch_size),1)
		concated = tf.concat(1,[indices,labels])
		onehot_labels = tf.sparse_to_dense(concated,[batch_size,vocab_size],1.0,0.0)

		logit_words = tf.matmul(output,out_W)+out_b
		output_id = tf.argmax(logit_words,1)

		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
		cross_entropy = cross_entropy * target_masks[:,i]

		loss = loss + tf.reduce_sum(cross_entropy)

	loss = loss / tf.reduce_sum(target_masks[:,1:])

	return loss

def inference(cell, state, out_W, out_b):
	outputs=[]
	for i in range(max_sequence_length):
		if(i==0):
			current_emb = tf.tile(first_dec_input,[1000,1])
		else:
			current_emb = tf.nn.embedding_lookup(embed_W,outputs[-1])+embed_b
		if i > 0 : tf.get_variable_scope().reuse_variables()
		# TODO Porpose of above line
		output,state = cell(current_emb,state)
		logit_words = tf.add(tf.matmul(output,out_W),out_b)

		output_id = tf.argmax(logit_words,1)
		outputs += [output_id]
	# print outputs

	return tf.transpose(tf.pack(outputs), perm=[1,0])


# Import data
def read_data(filename,testfilename,max_sequence_length):
	# Reading training file
	file = open(filename,'r')
	file_read = map(lambda x: ['GO']+x.strip().split(' ')+['EOW'],file.readlines())
	file.close()

	# Reading test file
	file = open(testfilename,'r')
	file_read_test = map(lambda x: ['GO']+x.strip().split(' ')+['EOW'],file.readlines())
	file.close()

	# Finding max sequence length of padding
	lengths = map(lambda x: len(x), file_read)
	lengths_test = map(lambda x: len(x), file_read_test)

	# max_sequence_length = max(max(lengths),max(lengths_test))
	num_words = len(lengths)
	
	char_to_id = defaultdict(lambda: len(char_to_id))

	#Padding test sequences
	file_read_test = map(lambda x: x+['PAD']*(max_sequence_length-len(x)),file_read_test)
	test_sequences = np.array([[char_to_id[char] for char in word] for word in file_read_test], dtype = np.int32)

	#Padding train sequences
	file_read = map(lambda x: x+['PAD']*(max_sequence_length-len(x)),file_read)
	sequences = np.array([[char_to_id[char] for char in word] for word in file_read], dtype = np.int32)
	masks = np.zeros([num_words, max_sequence_length],dtype = np.float32)
	for i in range(num_words):
		masks[i][:lengths[i]]=1
	
	return sequences, masks, lengths, max_sequence_length, char_to_id, test_sequences, lengths_test

def ids_to_word(sequence,id_to_word):
	return ' '.join(map(lambda x: id_to_word[x],sequence))

# embedding_size = 256
# batch_size = 256
# lrate = 0.001
import sys,os

# Reading args from command line
embedding_size,batch_size = map(int,(sys.argv)[1:3])
lrate = float(sys.argv[3])
rnn_size = embedding_size
infer_every = 50

# Creating output folder
output_folder = '_'.join(map(str,list([embedding_size,batch_size,lrate])))
if not os.path.exists(output_folder):
	os.makedirs(output_folder)


train_hi = 'data/train_hi'
test_hi = 'data/test_hi'
train_ka = 'data/train_ka'
test_ka = 'data/test_ka'

#Import data and create dictionaries for hindi and kannada
hi_train_sequences, hi_train_masks, hi_train_lengths, hi_max_sequence_length, hi_char_to_id, hi_test_sequences, hi_test_lengths = read_data(train_hi, test_hi,30)
hi_vocab_size = len(hi_char_to_id)
hi_id_to_char=dict()
for key in hi_char_to_id.keys():
	hi_id_to_char[hi_char_to_id[key]]=key

ka_train_sequences, ka_train_masks, ka_train_lengths, ka_max_sequence_length, ka_char_to_id, ka_test_sequences, ka_test_lengths = read_data(train_ka, test_ka,30)
ka_vocab_size = len(ka_char_to_id)
ka_id_to_char=dict()
for key in ka_char_to_id.keys():
	ka_id_to_char[ka_char_to_id[key]]=key

max_sequence_length = 30

max_val = 0.1

#Embedding

########Encoding

#feature vector size is 38 + 3 = 41
hi_phonetic_vecs = np.asarray([get_feature_vector(hi_id_to_char[i],'hi') for i in range(len(hi_id_to_char))])
ka_phonetic_vecs = np.asarray([get_feature_vector(ka_id_to_char[i],'kn') for i in range(len(ka_id_to_char))])

phonetic_vecs = tf.placeholder(shape = [None,41], dtype = tf.float32, name = 'phonetic_vecs')
# Matrix to multiply with
embed_W0 =tf.Variable(tf.random_uniform([41,embedding_size], -1*max_val, max_val), name = 'embed_W0') 

# Embedding size embeddings of hi and ka
# hi_embed_W = tf.matmul(hi_phonetic_vecs,embed_W0)
# ka_embed_W = tf.matmul(ka_phonetic_vecs,embed_W0)
embed_W = tf.matmul(phonetic_vecs,embed_W0)

#Same for both
embed_b = tf.Variable(tf.constant(0., shape=[embedding_size]), name = 'embed_b')

####Decoding
first_dec_input = tf.Variable(tf.random_uniform([1,embedding_size],dtype = tf.float32))

hi_out_b = tf.Variable(tf.constant(0., shape = [hi_vocab_size]))
hi_out_W = tf.Variable(tf.random_uniform([rnn_size,hi_vocab_size], -1*max_val, max_val))

ka_out_b = tf.Variable(tf.constant(0., shape = [ka_vocab_size]))
ka_out_W = tf.Variable(tf.random_uniform([rnn_size,ka_vocab_size], -1*max_val, max_val))

# Encoder Cell
enc_cell = rnn_cell.BasicLSTMCell(rnn_size)

#Decoder Cells
hi_dec_cell = rnn_cell.BasicLSTMCell(rnn_size)
ka_dec_cell = rnn_cell.BasicLSTMCell(rnn_size)

# Training batch placeholders
batch_sequences = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.int32)
batch_masks = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.float32)
batch_lengths = tf.placeholder(shape=[None],dtype=tf.float32)
# embed_W = tf.placeholder(shape=[None,embedding_size], dtype = tf.float32)

states = compute_hidden_representation(batch_sequences,batch_lengths,enc_cell,embed_W)

states_placeholder = tf.placeholder(shape=[None,2*rnn_size],dtype = tf.float32)
infer_output = inference(ka_dec_cell,states_placeholder,ka_out_W,ka_out_b)

learning_rate = tf.placeholder(shape=[],dtype=tf.float32)

hi_loss = seq_loss(batch_sequences,batch_masks,hi_dec_cell,hi_out_W,hi_out_b,states,hi_max_sequence_length, hi_vocab_size)
hi_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(hi_loss)

ka_loss = seq_loss(batch_sequences,batch_masks,ka_dec_cell,ka_out_W,ka_out_b,states,ka_max_sequence_length, ka_vocab_size)
ka_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ka_loss)

actual_words = map(lambda x: ids_to_word(x,ka_id_to_char), ka_test_sequences)
open(output_folder+'/actual_test','w').write('\n'.join(actual_words))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

hi_losses = []
ka_losses = []

for epoch in range(4000):
	if(epoch%2==0):
		batch_sequences1, batch_masks1, batch_lengths1 = get_batch(batch_size, hi_train_sequences, hi_train_masks, hi_train_lengths)
		l,_ = sess.run([hi_loss,hi_optimizer],feed_dict={batch_sequences: batch_sequences1, batch_masks: batch_masks1, batch_lengths: batch_lengths1, learning_rate: lrate, phonetic_vecs: hi_phonetic_vecs})
		hi_losses.append(l)
	else:
		batch_sequences1, batch_masks1, batch_lengths1 = get_batch(batch_size, ka_train_sequences, ka_train_masks, ka_train_lengths)
		l,_ = sess.run([ka_loss,ka_optimizer],feed_dict={batch_sequences: batch_sequences1, batch_masks: batch_masks1, batch_lengths: batch_lengths1, learning_rate: lrate, phonetic_vecs: ka_phonetic_vecs})
		ka_losses.append(l)

	if(epoch>0):
		print str(epoch).zfill(4)+'\thi: '+str(hi_losses[-1])+'\tka: '+str(ka_losses[-1])

	if((epoch+1)%infer_every == 0):
		states_ = sess.run(states,feed_dict={batch_sequences: hi_test_sequences, batch_lengths: hi_test_lengths, phonetic_vecs: hi_phonetic_vecs})
		predicted = sess.run(infer_output, feed_dict = {phonetic_vecs: ka_phonetic_vecs, states_placeholder: states_})
		predicted_words = map(lambda x: ids_to_word(x,ka_id_to_char), predicted)
		open(output_folder+'/epoch_'+(str(epoch+1)).zfill(4),'w').write('\n'.join(predicted_words))

# Write losses to a file
open(output_folder+'/hi_loss_output','w').write('\n'.join(map(str,hi_losses)))
open(output_folder+'/ka_loss_output','w').write('\n'.join(map(str,ka_losses)))