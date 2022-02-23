import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from os import system


datafp = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(datafp, "rb").read().decode(encoding="utf-8").lower()
text = text[300000:800000]
functions = ["boot"]
characters = sorted(set(text))

charidx = dict((c, i) for i, c in enumerate(characters))
idxchar = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3           # How many character to step to the next sentance.

sentences = []
next_chars = []

"""
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_chars.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), bool)
y = np.zeros((len(sentences), len(characters)), bool)

for i, sentence in enumerate(sentences):
    for t, char, in enumerate(sentence):
        x[i, t, charidx[char]] = 1
        y[i, charidx[next_chars[i]]] = 1


model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=4)
"""

model = tf.keras.models.load_model("inspire.model")

def sample(preds, temp=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def gen_text(length, temp=1.0):
    startidx = random.randint(0, len(text) - SEQ_LENGTH)
    generated = ""
    sentence = text[startidx:startidx+SEQ_LENGTH]
    generated += sentence

    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x[0, t, charidx[char]] = 1

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, temp)
        next_char = idxchar[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char

    return generated


system("clear")
print("Generating..\n")
print(gen_text(300, 0.2))
