import nltk

import json
# import pickle
# import random
import numpy as np
# import tensorflow as tf

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout 
from keras.optimizers.legacy import SGD 

# def define_network(X, y):
# 	tf.compat.v1.reset_default_graph() #Clears the default graph stack and resets the global default graph
# 	# neural network's layers
# 	network = tflearn.input_data(shape= [None, len(X[0])]) #input layer
# 	network = tflearn.fully_connected(network, 8) #1st hidden layer
# 	network = tflearn.fully_connected(network, 8) #2nd hidden layer
# 	network = tflearn.fully_connected(network, len(y[0]), activation= 'softmax') #output layer
# 	network = tflearn.regression(network)
# 	model = tflearn.DNN(network, tensorboard_dir='tflearn_logs') #tensorboard_dir is path to store logs
# 	return model


# get root word of any words Dibby doesn't know
lemmatizer = WordNetLemmatizer()

# load file of intents for processing
intents = json.loads(open("intent.json").read())

# creating empty lists to store contents of intents ****
# words contains all words in the patterns lists
words = [] 
# classes contans all tags
classes = []
# documents contains the patterns and their associated tags
documents = [] 

# ignore punctuation
ignore_letter = ["?", "!", ".", ","]

# iterate through all intents
for intent in intents['intents']:
    # iterate through all patterns for a single intent
    for pattern in intent['patterns']:
        # word_list contains all words in the "patterns" list
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # associate the patterns w/ their respective tags
        documents.append((word_list, intent['tag']))
        # append tags to the class list if not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# modify words list so it only contans the root words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letter]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# # save words & classes list to bnary files
# pickle.dump(words, open('words.pk1', 'wb'))
# pickle.dump(classes, open('classes.pk1', 'wb'))

# neural networks work w/ numerical values, not strings
    # classify data (words) as 0's and 1's
# training will store data (patterns) used for training
training = []
# output will store data (classes) used for training
output = []
# output_empty will store a 0 for every class in intent.json
output_empty = [0]*len(classes)
for document in documents:
    # bag contans 0 if a word in words isn't in the pattern, 1 if it is
    bag = []
    # list of words associated w/ certain tag
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    # output row indicates which class is associated with the current "bag" (patterns kinda)
    output_row[classes.index(document[1])] = 1
    training.append(bag)
    output.append(output_row)

# make data a numpy array
training = np.array(training)
output = np.array(output)

train_x = list(training)
train_y = list(output)
print("Yay1!") 

# creating a Sequential machine learning model 
model = Sequential() 
model.add(Dense(128, input_shape=(len(train_x[0]), ), 
				activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(len(train_y[0]), 
				activation='softmax')) 

# compiling the model 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy', 
			optimizer=sgd, metrics=['accuracy']) 
hist = model.fit(np.array(train_x), np.array(train_y), 
				epochs=200, batch_size=5, verbose=1) 

# saving the model 
model.save("chatbotmodel.h5", hist) 

# print statement to show thesuccessful training of the Chatbot model 
print("Yay2!") 
