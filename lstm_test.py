#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM for text generation
based on keras demo https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

"""
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

from load_data import shakespeare_text, char_to_int, int_to_char
nchar = len(shakespeare_text)
ndim = len(char_to_int.keys())

#number of characters seen at one time
maxlen = 20
#how much the strings overlap. step=1 means only 1 one character advanced between strings
step=1

text=shakespeare_text

strs = []
next_chars =[]
for i in range(0,nchar-maxlen, step):

    strs.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

nstr=len(strs)
print("Number of strings ", nstr)

#vectorize
# the shape of the lstm input includes two dimensions per sample
x = np.zeros((nstr, maxlen, ndim))
y = np.zeros((nstr, ndim))

print("Vectorizing. Input shape:")
print(x.shape)
for i in range(nstr):
    for j in range(maxlen):
        x[i, j, char_to_int[strs[i][j]]]=1
    y[i, char_to_int[next_chars[i]]]=1


#number of hidden units
nh=100

print("Building model")
model = Sequential()
model.add(LSTM(nh, input_shape=(maxlen, ndim)))
model.add(Dense(ndim))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', 
                  loss= 'categorical_crossentropy', 
                  metrics=['accuracy'])

from keras.callbacks import LambdaCallback
import time
class Watch:
    def __init__(self):
        self.t = time.time()
    def tick(self):
        t=time.time()
        dt = t - self.t
        self.t = t
        return dt
        
watch = Watch()


    
#sample, given a particular distribution of probabilities
# and some effective relative temp
# that is, define the given probabilities are thermal -- then rescale them
# by the given temp

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = preds.flatten().astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    #let numpy do the sampling
    return np.argmax(np.random.multinomial(1, preds, 1))
    
    
def to_x(seq):
    x = np.zeros((1, maxlen, ndim))
    for i in range(maxlen):
        x[0, i, char_to_int[seq[i]]]=1
    return x
    

###generate a text sequence based on random init!
def gen_string(model, seed_string, nchar, temp=1.0):
    string_pred=seed_string
    seq=seed_string
    i= np.random.randint(ndim)
    for _ in range(nchar):
        x = to_x(seq)
        #the vector of output probabilities
        p = model.predict(x)
        #predict a character
        ipred = sample(p, temperature=temp)
        cpred = int_to_char[ipred]
        #update
        string_pred += cpred
        seq = seq[1:] + cpred
    return string_pred


def on_epoch_end(epoch, logs):
    print("epoch end -- ", epoch)
    print("time: {0:.3f}s".format(watch.tick()))
    #generate text
    i0 = np.random.randint(nchar//2)
    seed = text[i0:i0+maxlen]
    s=gen_string(model, seed, 50,temp=.1)
    print("Sample gen:")
    print(s)
    
callback = LambdaCallback(on_epoch_end=on_epoch_end)

print("Training")
model.fit(x, y, 
              batch_size=100, 
                epochs=50, 
                callbacks=[callback])

#generate text

i0 = np.random.randint(nchar//2)
seed = text[i0:i0+maxlen]
s=gen_string(model, seed, 200, .2)
print(s)

def truncate(s):
    return s.lower()[:maxlen]

seed = truncate("My car is running on fumes")
s= gen_string(model, seed, 200, .2)
print(s)