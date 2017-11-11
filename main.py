# Import all libraries

from __future__ import print_function
from __future__ import division
import os
import argparse
import torch
import torch.utils.data

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import pytorch_crepe
import datetime
import numpy as np
import data_helpers
from CharCNN import CharCNN
import csv
np.random.seed(0123)


# set parameters:

subset = None


#Compile/fit params
batch_size = 50
nb_epoch = 10
maxlen = 256



print('Loading data...')
#Expect x to be a list of sentences. Y to be a one-hot encoding of the
#categories.
(xt, yt), (x_test, y_test) = data_helpers.load_ag_data()

print('Creating vocab...')
vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()
test_data = data_helpers.encode_data(x_test, maxlen, vocab, vocab_size, check)



print('Build model...')

model = CharCNN()
model.cuda()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01,momentum=0.9)



print('train model...')


test_final_loss=[]
test_final_accuracy=[]

for e in xrange(nb_epoch):
    xi, yi = data_helpers.shuffle_matrix(xt, yt)
    xi_test, yi_test = data_helpers.shuffle_matrix(x_test, y_test)
    if subset:
        batches = data_helpers.mini_batch_generator(xi[:subset], yi[:subset],
                                                    vocab, vocab_size, check,
                                                    maxlen,
                                                    batch_size=batch_size)
    else:
        batches = data_helpers.mini_batch_generator(xi, yi, vocab, vocab_size,
                                                    check, maxlen,
                                                    batch_size=batch_size)

    test_batches = data_helpers.mini_batch_generator(xi_test, yi_test, vocab,
                                                     vocab_size, check, maxlen,
                                                     batch_size=1)

    accuracy = 0.0
   
    step = 1
    start = datetime.datetime.now()
    print('Epoch: {}'.format(e))
    running_loss = 0.0
    i=0
    train_loss_avg=0.0
    train_correct=0.0
    for x_train, y_train in batches:
        i=i+1
        inputs=torch.from_numpy(np.swapaxes(x_train.astype(np.float64),1,2))
        labels=torch.from_numpy(np.argmax(y_train, axis=1).astype(np.float64))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        inputs=inputs.float()
        labels=labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #train_loss += loss.data[0]
        train_loss_avg =loss.data.mean()
        _, predicted = torch.max(outputs.data, 1)
        train_correct = (predicted == labels.data).sum()
        accuracy +=  (train_correct*100/len(labels))
        accuracy_avg = accuracy / step
        if step % 200 == 0:
            print('  Step: {} Training Loss: {}. Training accuracy: {}'.format(step,train_loss_avg,accuracy_avg))
        step += 1
    test_accuracy=0
    test_loss_avg=0
    test_correct=0
    step=1
    test_accuracy_avg=0
    for x_test_batch, y_test_batch in test_batches:
        inputs=torch.from_numpy(np.swapaxes(x_test_batch.astype(np.float64),1,2))
        labels=torch.from_numpy(np.argmax(y_test_batch, axis=1).astype(np.float64))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        inputs=inputs.float()
        labels=labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss_avg += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        test_correct = (predicted == labels.data).sum()
        test_accuracy +=  (test_correct*100/len(labels))
        #test_accuracy_avg += (test_accuracy /step)
        step+=1
    print('Test Loss: {}. Test Accuracy: {}'.format(test_loss_avg/step,test_accuracy/step))
    test_final_loss.append(test_loss_avg/step)
    test_final_loss.append(test_accuracy/step)


with open('test_loss.txt', 'wb') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(test_final_loss)
with open('test_accuracy.txt', 'wb') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(test_final_accuracy)
