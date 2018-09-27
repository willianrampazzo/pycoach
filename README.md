# PyCoach: a training package for PyTorch

PyCoach is a Python package that provides:
- A high-level abstraction for PyTorch model training;
- Well known callbacks structure based on Keras;

The main purpose of PyCoach is handle training, evaluation and prediction, leaving other tasks like create the network model and dataloaders to the users.

This is an initial personal project and contribution to it is appreciated.

Actual status: Working with PyTorch 0.4!

## To Do:

- Document the use of the package;
- Add use examples;
- Document the code;
- Complete Keras callbacks ports to PyCoach:
    - [x] CallbackList
    - [x] Callback
    - [x] BaseLogger
    - [x] TerminateOnNaN
    - [x] ProgbarLogger
    - [x] History
    - [x] ModelCheckpoint
    - [x] EarlyStopping
    - [x] RemoteMonitor
    - [ ] LearningRateScheduler
    - [ ] TensorBoard
    - [ ] ReduceLROnPlateau
    - [x] CSVLogger
    - [x] LambdaCallback
    - [x] Plotter (new!)
    - [x] TextLogger (new!)
- Improve predict function;

## Links:

- [PyTorch](https://github.com/pytorch/pytorch)
- [Keras](https://github.com/fchollet/keras)

