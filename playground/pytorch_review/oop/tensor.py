# MINI PYTORCH PRACTICE PROJECT (numpy only)
#
# Make a Tensor class that wraps a numpy array and has a .grad attribute.
# Give it the dunder methods __add__, __mul__, __matmul__, __neg__, __sub__,
# and __pow__, so you can write expressions like (x @ W + b) ** 2 directly.
# Each operation should remember its inputs and how to compute their
# gradients. Add a backward() method that walks back through those inputs
# and fills in .grad (basic autograd). Add __repr__ so tensors print
# readably.
#
# Make an abstract base class Module that inherits from ABC. Give it an
# abstract method forward. Give it a normal __call__ that just calls
# self.forward(...), so model(x) works the same way it does in PyTorch. Give
# it two inherited methods: parameters(), which yields every Tensor stored on
# the module, including those inside child modules, and zero_grad(), which
# resets their gradients.
#
# Make Linear(Module) and ReLU(Module). Both of their __init__ methods must
# call super().__init__() before setting anything else up.
#
# Make Sequential(Module), which takes a list of modules and runs them in
# order. Give it __len__, __getitem__, and __iter__ so you can index into a
# model and loop over its layers.
#
# Make an abstract Dataset class with abstract __len__ and __getitem__
# methods. Subclass it as a ToyDataset that generates points from y = 3x + 2
# plus some noise. Make a DataLoader class whose __iter__ yields shuffled
# batches.
#
# Practice multiple inheritance with mixins: make a LoggingMixin that prints
# the input and output shapes in forward and calls super().forward(). Then
# make class LoggedLinear(LoggingMixin, Linear). Predict its __mro__ before
# you check it.
#
# Make an abstract Optimizer class with an abstract step method and a
# concrete zero_grad method. Subclass it as SGD.
#
# Train a small Sequential model on the ToyDataset with an MSE loss written
# out of Tensor operations. Check that the loss goes down and that the
# learned weights end up near 3 and 2.
#
# Add type hints everywhere and run mypy. Confirm that Module() raises a
# TypeError.