import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
from matplotlib import pyplot as plt
import os
import re

# Dataset used here can be found at https://www.kaggle.com/aiswaryaramachandran/hindienglish-corpora
# Download the dataset and put the path in read_csv() function accordingly
data = pd.read_csv("./Hindi_English_Truncated_Corpus.csv")
english_sentences = data["english_sentence"]
hindi_sentences = data["hindi_sentence"]

num_words = 10000
oov_token = '<UNK>'
english_vocab_size = num_words + 1
hindi_vocab_size = num_words + 1
MAX_WORDS_IN_A_SENTENCE = 16
test_ratio = 0.2
BATCH_SIZE = 512
embedding_dim = 64
hidden_units = 1024
learning_rate = 0.006
epochs = 100

def preprocess_sentence(sen, is_english):
	if (type(sen) != str):
		return ''
	sen = sen.strip('.')
	
	# insert space between words and punctuations
	sen = re.sub(r"([?.!,¿;।])", r" \1 ", sen)
	sen = re.sub(r'[" "]+', " ", sen)
	
	# For english, replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
	if(is_english == True):
		sen = re.sub(r"[^a-zA-Z?.!,¿']+", " ", sen)
		sen = sen.lower()
	
	sen = sen.strip()
	sen = 'sentencestart ' + sen + ' sentenceend'
	
	sen = ' '.join(sen.split())
	return sen
	
# Loop through each datapoint having english and hindi sentence
processed_e_sentences = []
processed_h_sentences = []
for (e_sen, h_sen) in zip(english_sentences, hindi_sentences):
	processed_e_sen = preprocess_sentence(e_sen, True)
	processed_h_sen = preprocess_sentence(h_sen, False)
	if(processed_e_sen == '' or processed_h_sen == '' or processed_e_sen.count(' ') >  (MAX_WORDS_IN_A_SENTENCE-1) or processed_h_sen.count(' ') > (MAX_WORDS_IN_A_SENTENCE-1)):
		continue
	
	processed_e_sentences.append(processed_e_sen)
	processed_h_sentences.append(processed_h_sen)

print("Sentence examples: ")
print(processed_e_sentences[0])
print(processed_h_sentences[0])
print("Length of English processed sentences: " + str(len(processed_e_sentences)))
print("Length of Hindi processed sentences: " + str(len(processed_h_sentences)))

def tokenize_sentences(processed_sentences, num_words, oov_token):
	tokenizer = Tokenizer(num_words = num_words, oov_token = oov_token)
	tokenizer.fit_on_texts(processed_sentences)
	word_index = tokenizer.word_index
	sequences = tokenizer.texts_to_sequences(processed_sentences)
	sequences = pad_sequences(sequences, padding = 'post')
	return word_index, sequences, tokenizer

english_word_index, english_sequences, english_tokenizer = tokenize_sentences(processed_e_sentences, num_words, oov_token)
hindi_word_index, hindi_sequences, hindi_tokenizer = tokenize_sentences(processed_h_sentences, num_words, oov_token)

# split into traning and validation set
english_train_sequences, english_val_sequences, hindi_train_sequences, hindi_val_sequences = train_test_split(english_sequences, hindi_sequences, test_size = test_ratio)
BUFFER_SIZE = len(english_train_sequences)

# Batching the training set
dataset = tf.data.Dataset.from_tensor_slices((english_train_sequences, hindi_train_sequences)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)
print("No. of batches: " + str(len(list(dataset.as_numpy_iterator()))))



class Encoder(tf.keras.Model):
	
	def __init__(self, english_vocab_size, embedding_dim, hidden_units):
		super(Encoder, self).__init__()
		self.embedding = tf.keras.layers.Embedding(english_vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(hidden_units, return_sequences = True, return_state = True)
		
	def call(self, input_sequence):
		x = self.embedding(input_sequence)
		encoder_sequence_output, final_encoder_state = self.gru(x)
		#	Dimensions of encoder_sequence_output => (BATCH_SIZE, MAX_WORDS_IN_A_SENTENCE, hidden_units)
		#	Dimensions of final_encoder_state => (BATCH_SIZE, hidden_units)
		return encoder_sequence_output, final_encoder_state

# initialize our encoder
encoder = Encoder(english_vocab_size, embedding_dim, hidden_units)

class BasicDotProductAttention(tf.keras.layers.Layer):
	def __init__(self):
		super(BasicDotProductAttention, self).__init__()
		
	def call(self, decoder_hidden_state, encoder_outputs):
		#	Dimensions of decoder_hidden_state => (BATCH_SIZE, hidden_units)
		#	Dimensions of encoder_outputs => (BATCH_SIZE, MAX_WORDS_IN_A_SENTENCE, hidden_units)

		decoder_hidden_state_with_time_axis = tf.expand_dims(decoder_hidden_state, 2)
		#	Dimensions of decoder_hidden_state_with_time_axis => (BATCH_SIZE, hidden_units, 1)
		attention_scores = tf.matmul(encoder_outputs, decoder_hidden_state_with_time_axis)
		#	Dimensions of attention_scores => (BATCH_SIZE, MAX_WORDS_IN_A_SENTENCE, 1)
		attention_scores = tf.nn.softmax(attention_scores, axis = 1)
		weighted_sum_of_encoder_outputs = tf.reduce_sum(encoder_outputs * attention_scores, axis = 1)
		#	Dimensions of weighted_sum_of_encoder_outputs => (BATCH_SIZE, hidden_units)

		return weighted_sum_of_encoder_outputs, attention_scores

class Decoder(tf.keras.Model):
	def __init__(self, hindi_vocab_size, embedding_dim, hidden_units):
		super(Decoder, self).__init__()
		self.embedding = tf.keras.layers.Embedding(hindi_vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(hidden_units, return_state = True)
		self.word_probability_layer = tf.keras.layers.Dense(hindi_vocab_size, activation = 'softmax')
		self.attention_layer = BasicDotProductAttention()
		
	def call(self, decoder_input, decoder_hidden, encoder_sequence_output):
		
		x = self.embedding(decoder_input)
		#	Dimensions of x => (BATCH_SIZE, embedding_dim)
		weighted_sum_of_encoder_outputs, attention_scores = self.attention_layer(decoder_hidden, encoder_sequence_output)
		#	Dimensions of weighted_sum_of_encoder_outputs => (BATCH_SIZE, hidden_units)
		x = tf.concat([weighted_sum_of_encoder_outputs, x], axis = -1)
		x = tf.expand_dims(x, 1)
		#	Dimensions of x => (BATCH_SIZE, 1, hidden_units + embedding_dim)
		decoder_output, decoder_state = self.gru(x)
		#	Dimensions of decoder_output => (BATCH_SIZE, hidden_units)
		word_probability = self.word_probability_layer(decoder_output)
		#	Dimensions of word_probability => (BATCH_SIZE, hindi_vocab_size)
		return word_probability, decoder_state, attention_scores

# initialize our decoder
decoder = Decoder(hindi_vocab_size, embedding_dim, hidden_units)


optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
def loss_function(actual_words, predicted_words_probability):
	loss = loss_object(actual_words, predicted_words_probability)
	mask = tf.where(actual_words > 0, 1.0, 0.0)
	return tf.reduce_mean(mask * loss)

def train_step(english_sequences, hindi_sequences):
	loss = 0
	with tf.GradientTape() as tape:
		encoder_sequence_output, encoder_hidden = encoder(english_sequences)
		decoder_hidden = encoder_hidden
		decoder_input = hindi_sequences[:, 0]
		for i in range(1, hindi_sequences.shape[1]):
			predicted_words_probability, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_sequence_output)
			actual_words = hindi_sequences[:, i]
			# if all the sentences in batch are completed
			if np.count_nonzero(actual_words) == 0:
				break
			loss += loss_function(actual_words, predicted_words_probability)

			decoder_input = actual_words

	variables = encoder.trainable_variables + decoder.trainable_variables
	gradients = tape.gradient(loss, variables)
	optimizer.apply_gradients(zip(gradients, variables))
	return loss.numpy()

all_epoch_losses = []
training_start_time = time.time()
for epoch in range(epochs):
	epoch_loss = []
	start_time = time.time()
	for(batch, (english_sequences, hindi_sequences)) in enumerate(dataset):
		batch_loss = train_step(english_sequences, hindi_sequences)
		epoch_loss.append(batch_loss)

	all_epoch_losses.append(sum(epoch_loss)/len(epoch_loss))
	print("Epoch No.: " + str(epoch) + " Time: " + str(time.time()-start_time))

print("All Epoch Losses: " + str(all_epoch_losses))
print("Total time in training: " + str(time.time() - training_start_time))

plt.plot(all_epoch_losses)
plt.xlabel("Epochs")
plt.ylabel("Epoch Loss")
plt.show()

def get_sentence_from_sequences(sequences, tokenizer):
	return tokenizer.sequences_to_texts(sequences)

# Testing
def translate_sentence(sentence):
	sentence = preprocess_sentence(sentence, True)
	sequence = english_tokenizer.texts_to_sequences([sentence])[0]
	sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen = MAX_WORDS_IN_A_SENTENCE, padding = 'post')
	encoder_input = tf.convert_to_tensor(sequence)
	encoder_sequence_output, encoder_hidden = encoder(encoder_input)
	decoder_input = tf.convert_to_tensor([hindi_word_index['sentencestart']])
	decoder_hidden = encoder_hidden
	
	sentence_end_word_id = hindi_word_index['sentenceend']
	hindi_sequence = []
	for i in range(MAX_WORDS_IN_A_SENTENCE*2):
		predicted_words_probability, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_sequence_output)
		# taking the word with maximum probability
		predicted_word_id = tf.argmax(predicted_words_probability[0]).numpy()
		hindi_sequence.append(predicted_word_id)
		# if the word 'sentenceend' is predicted, exit the loop
		if predicted_word_id == sentence_end_word_id:
			break
		decoder_input = tf.convert_to_tensor([predicted_word_id])
	print(sentence)
	return get_sentence_from_sequences([hindi_sequence], hindi_tokenizer)

# print translated sentence
print(translate_sentence("Write a sentence here to check how good model is working!"))

		




