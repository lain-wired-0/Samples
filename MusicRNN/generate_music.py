import tensorflow as tf
import numpy as np
import os
import time
from tqdm import tqdm

from tools_api import *

# get songs dataset
songs = load_training_data()    # list of all songs

# look at example song
example_song = songs[0]
print('Example song:')
print(examples_song)

# join list of all songs to a string
songs_joined = '\n\n'.join(songs)
vocab = sorted(set(songs_joined))   # all unique characters from all songs
print('There are', len(vocab), 'unque characters in the dataset')

# numerical representation of text
# conversion from unique character to a number
char2idx = {u:i for i, u in enumerate(vocab)}
# conversion of number back to unique character; number is the index of the array to get the character back, could also use a dict for easier understanding but this is more efficient
idx2char = np.array(vocab)

# convert songs string to numerical representation
def vectorize_string(string):
    return np.array([char2idx(character) for character in string])
vectorized_songs = vectorize_string(songs_joined)

# get batches of training examples
def get_batch(vectorized_songs, seq_length, batch_size):
    # length of vectorized_songs
    n = vectorized_songs.shape[0] - 1
    # array of random starting indices to the vectorized song string of size batch_size
    # ex: if batch size is 4, there would be 4 random starting indices
    idx = np.random.choice(n - seq_length, batch_size)
    # input sequence
    input_batch = [vectorized_songs[i:i + seq_length] for i in idx]
    # matching output sequence
    output_batch = [vectorized_songs[i + 1:i + seq_length + 1] for i in idx]
    # reshape into array so that each index is a new snippet 
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch

# define lstm
def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True)
# rnn model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        # embedding layer: transform char2idx indices to dense vectors of fixed embedding size
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        # ltsm layer
        # rnn_units: Positive integer, dimensionality of the output space.
        LTSM(rnn_units),
        # dense layers: fully connected layer that transformas lstm output into vocabulary size
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# loss function
def compute_loss(labels, logits):
    # labels: true targets
    # logits: predicted targets
    # sparse_categorical_cross_entropy: input true and predicted indices and will output corresponding loss values
    loss = tf.keras.losses.sparse_categorical_cross_entropy(labels, logits, from_logits=True)
    return loss

# hyperparameter settings
# Optimization parameters:
num_training_iterations = 2000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1
# Model parameters: 
vocab_size = len(vocab)
embedding_dim = 256 
rnn_units = 1024  # Experiment between 1 and 2048
# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

# training
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # model pred labels
        y_hat = model(x)
        # compute loss
        loss = compute_loss(y, y_hat)
    # compute gradients with respect to all model parameters
    grads = tape.gradients(loss, model.trainable_variables)
    # apply gradients to update model
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
history = []
plotter = PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='loss')
if hasattr(tdqm, '_instances'):
    # clear is it exits
    tqdm._instances.clear()
for iter in tqdm(range(num_training_iterations)):
    # get batch and propogate through network
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
    loss = train_step(x_batch, y_batch)
    # update progress bar
    history.append(loss.numpy().mean())
    plotter.plot(history)
    # update model with changed weights
    if iter % 100 == 0:
        model.save_weights(checkpoint_prefix)
# save trained model and weights
model.save_weights(checkpoint_prefix)

# inference
# rebuild model with batch_size=1
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
# restore latest weights
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

# generate new song
def generate_test(model, start_string, generation_length=1000):
    # convert start string to numbers
    input_eval = [vectorize_string(start_string)]
    input_eval = tf.expand_dims(input_eval, 0)

    # empty string to store results
    text_generated = []

    # batch_size == 1
    # starting model again so reset lstm state bc we want to start it new
    model.reset_states()
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        # output prediction is length seq_length
        predictions = model(input_eval)
        # remove batch dimension
        predictions = tf.squeeze(predictions, 0)
        # sample from multinomial distribution of to calculate index of predicted character,
        #   this predicted character will be next input into model
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # pass prediction and previous hidden state as next inputs to model
        input_eval = tf.expand_dims([predicted_id], 0)
        # add prediction to output text
        text_generated.append(idx2char[predicted_id])
    
    return (start_string + ''.join(text_generated))

# create song
# abc files start with 'X'
generated_text = generate_text(model, start_string='X', generation_length=1000)

generated_songs = extract_song_snippet(generated_text)
for i, song in enumerate(generated_songs):
    waveform = play_song(song)
