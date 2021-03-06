""" Coach Module """
#from __future__ import absolute_import

import warnings
import csv
import torch

from . import callbacks as cbks

class Coach:
    """
        Coach class
    """

    def __init__(self, model, loaders, optimizer=None, loss_fn=None,
                 use="Auto"
                ):
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        if use == "Auto":
            use = torch.cuda.is_available()
            self.device = torch.device("cuda" if use else "cpu")
        else:
            self.device = use

        self.history = None
        self.stop_training = False
    # __init__()

    def evaluate(self, loader, loss_fn=None):
        """
            Evaluation
        """
        if not loss_fn:
            loss_fn = self.loss_fn

        # local variables
        total_loss = 0
        batches = 0

        self.model.eval()
        with torch.no_grad():
            for batches, data in enumerate(loader):
                if isinstance(data, dict):
                    inputs, labels = data.values()
                else:
                    inputs, labels = data

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass
                outputs = self.model(inputs)

                # loss
                loss = loss_fn(outputs, labels)

                # metrics
                total_loss += loss.item()
        self.model.train()

        total_loss = total_loss / batches
        return total_loss
    # evaluation()

    def load(self, fname, weights_only=False):
        """
            Load Model
        """
        if weights_only:
            self.model.load_state_dict(
                torch.load(fname,
                           map_location=torch.device(self.device)))
        else:
            self.model = torch.load(fname,
                                    map_location=torch.device(self.device))
    # load()

    def predict(self, loader):
        """
            Predict
        """
        outputs = torch.Tensor()
        outputs = outputs.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for data in loader:
                if isinstance(data, dict):
                    inputs, labels = data.values()
                else:
                    inputs, labels = data

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass
                outputs = torch.cat((outputs, self.model(inputs).data))
        self.model.train()
        return outputs
    # predict()

    def save(self, fname, weights_only=False):
        """
            Save Model
        """
        if weights_only:
            torch.save(self.model.state_dict(), fname)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.save(self.model, fname)
    # save()

    def save_history(self, fname, append=False):
        """
            Save Training History
        """
        if not self.history.history:
            raise ValueError('No history available to save!')

        if append:
            mode = 'a'
        else:
            mode = 'w'

        header = sorted([k for k in self.history.history])
        header.insert(0, 'epoch')

        data = [self.history.epoch]
        for k in range(1, len(header)):
            data.append(self.history.history[header[k]])
        data = list(map(list, zip(*data)))

        with open(fname, mode) as csvfile:
            w = csv.writer(csvfile)
            if not append:
                w.writerow(header)
            w.writerows(data)
    # save_history()

    def train(self, epochs, optimizer=None, loss_fn=None, callbacks=None,
              verbose=1,
             ):
        """
            Train
        """

        if not optimizer and self.optimizer is not None:
            optimizer = self.optimizer
        else:
            raise ValueError('You should set an optimizer before ',
                             'train a network!')

        if not loss_fn and self.loss_fn is not None:
            loss_fn = self.loss_fn
        else:
            raise ValueError('You should set a loss function before ',
                             'train a network!')

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
                print('Train on %d samples' % (num_train_samples))

        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
        callbacks = cbks.CallbackList(callbacks)

        callbacks.set_model(self)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
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

                    inputs, labels = inputs.to(self.device), \
                                     labels.to(self.device)

                    # zeroes gradients
                    optimizer.zero_grad()

                    # forward pass
                    predict = self.model(inputs)

                    # loss
                    loss = loss_fn(predict, labels)

                    # calculate new gradients
                    loss.backward()

                    # update weights
                    optimizer.step()

                    # metrics
                    batch_logs['loss'] = loss.item()
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
