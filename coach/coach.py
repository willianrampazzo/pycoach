""" Coach Module """
from __future__ import absolute_import

#import numpy as np
import warnings
import torch

#from copy import deepcopy
#from time import time
from torch.autograd import Variable

from . import callbacks as cbks

class Coach:
    """
        PyCoach class
    """

    def __init__(self,
                 model,
                 loaders,
                 optimizer=None,
                 loss_fn=None
                ):
        """

        """
        self.use_gpu = torch.cuda.is_available()
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.history = None
        self.stop_training = False
    # __init__()

    def evaluate(self, loader, loss_fn=None):
        """
            Evaluation Method
        """
        if not loss_fn:
            loss_fn = self.loss_fn

        # local variables
        total_loss = 0
        batches = 0

        self.model.eval()
        for batches, data in enumerate(loader):
            if isinstance(data, dict):
                inputs, labels = data.values()
            else:
                inputs, labels = data

            if self.use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward pass
            outputs = self.model(inputs)

            # loss
            loss = loss_fn(outputs, labels)

            # metrics
            total_loss += loss.data[0]
        self.model.train()

        total_loss = total_loss / batches
        return total_loss
    # evaluation()

    def load(self, fname, weights_only=False):
        """
            Load Method
        """
        if weights_only:
            self.model.load_state_dict(torch.load(fname))
        else:
            self.model = torch.load(fname)
    # load()

    def predict(self, loader):
        """
            Predict Method
        """
        #outputs = loader.dataset.data_tensor.new()
        outputs = torch.Tensor()
        if self.use_gpu:
            outputs = outputs.cuda()
        self.model.eval()
        for data in loader:
            if isinstance(data, dict):
                inputs, labels = data.values()
            else:
                inputs, labels = data
            if self.use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # forward pass
            outputs = torch.cat((outputs, self.model(inputs).data))
        self.model.train()
        return outputs
    # predict()

    def save(self, fname, weights_only=False):
        """
            Save Method
        """
        if weights_only:
            torch.save(self.model.state_dict(), fname)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.save(self.model, fname)
    # save()

    def train(self,
              epochs,
              optimizer=None,
              loss_fn=None,
              callbacks=None,
              verbose=1,
             ):
        """
            Train Method
        """

        if not optimizer and self.optimizer is not None:
            optimizer = self.optimizer
        else:
            raise ValueError('You should set an optimizer before ',
                             'train a network!')

        if not loss_fn:
            loss_fn = self.loss_fn

        num_train_samples = len(self.loaders['train'].dataset)
        batch_size = self.loaders['train'].batch_size

        do_validation = False
        if len(self.loaders) > 1:
            do_validation = True
            callback_metrics = ['loss', 'val_loss']
            num_val_samples = len(self.loaders['validate'].dataset)

            if verbose > 0:
                print('Train on %d samples, validate on %d samples' %
                      (num_train_samples, num_val_samples))
        else:
            callback_metrics = ['loss']
            if verbose > 0:
                print('Train on %d samples' %
                      (num_train_samples))

        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
        if verbose:
            callbacks.insert(1, cbks.ProgbarLogger())
        callbacks = cbks.CallbackList(callbacks)

        callbacks.set_model(self)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            #'steps': steps_per_epoch,
            'samples': num_train_samples,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics or [],
        })
        callbacks.on_train_begin()
        self.stop_training = False

        try:
            for epoch in range(epochs):
                callbacks.on_epoch_begin(epoch)
                epoch_logs = {}

                for batch, data in enumerate(self.loaders['train']):
                    batch_logs = {}
                    batch_logs['batch'] = batch
                    batch_logs['size'] = batch_size
                    callbacks.on_batch_begin(batch, batch_logs)
                    # data inputs
                    if isinstance(data, dict):
                        inputs, labels = data.values()
                    else:
                        inputs, labels = data

                    if self.use_gpu:
                        inputs, labels = Variable(inputs.cuda()), \
                                Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # forward pass
                    predict = self.model(inputs)

                    # loss
                    loss = loss_fn(predict, labels)

                    # zeroes gradients
                    optimizer.zero_grad()

                    # calculate new gradients
                    loss.backward()

                    # update weights
                    optimizer.step()

                    # metrics
                    batch_logs['loss'] = loss.data[0]
                    callbacks.on_batch_end(batch, batch_logs)
                    if self.stop_training:
                        break

                # loss on validation set
                if do_validation:
                    epoch_logs['val_loss'] = (self.evaluate(
                        self.loaders['validate'],
                        loss_fn))
                callbacks.on_epoch_end(epoch, epoch_logs)
                if self.stop_training:
                    break

        except  KeyboardInterrupt:
            if verbose > 0:
                print('Training interrupted!')

        callbacks.on_train_end()
        if verbose > 0:
            print('Training completed!')
        return self.history
    # train()
