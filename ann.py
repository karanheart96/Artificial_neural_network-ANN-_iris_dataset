# first neural network with keras tutorial
from numpy import loadtxt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
# load the dataset
dataset = loadtxt('Iris.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:4]
y = dataset[:,4]
y = to_categorical(y)
# define the keras model
model = Sequential()
model.add(Dense(50, input_dim=4, activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(3, activation='softmax'))
# compile the keras model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=20, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))