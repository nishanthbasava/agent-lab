# MINI MICROGRAD PRACTICE PROJECT (pure Python, scalars only, PyTorch names)
#
# Make a Value class that wraps a single number. Give it .data, .grad
# (starts at 0.0), _prev (the Values it was made from), and _backward (a
# function that passes its gradient to its parents; it does nothing by
# default). Add __repr__ so Values print readably.
#
# Give Value the dunder methods __add__, __mul__, __pow__ (number exponent
# only), __neg__, __sub__, and __truediv__. Also add __radd__, __rmul__, and
# __rsub__ so expressions like 2 * x and 1 + x work. If the other operand is
# a plain number, wrap it in a Value first. Each operation returns a new
# Value with _prev and _backward set, and gradients always use += not =.
#
# Add the methods relu() and tanh().
#
# Add a backward() method. It topologically sorts the graph starting from
# this Value, sets this Value's grad to 1.0, then calls _backward() on every
# node in reverse order. Test it by hand on L = a*b + a and check that
# a.grad == b + 1 and b.grad == a.
#
# Make an abstract base class Module that inherits from ABC. Give it an
# abstract method forward and a normal __call__ that calls
# self.forward(...), so model(x) works. Give it a parameters() method that
# returns a list of Values and defaults to an empty list, and a zero_grad()
# method that sets every parameter's grad to 0.
#
# Make Neuron(Module). It holds a list of weight Values and one bias Value,
# and its forward takes a list of numbers or Values and returns
# sum(w*x) + b as a single Value. It has no activation.
#
# Make Linear(Module). Its __init__ takes in_features and out_features and
# creates out_features Neurons, each with in_features weights. Its forward
# takes a list of Values and returns a list with one output per Neuron.
# Its parameters() collects every Neuron's parameters.
#
# Make ReLU(Module). It has no parameters, and its forward returns
# [v.relu() for v in x].
#
# Make Sequential(Module). It takes a list of modules and runs them in
# order, feeding each one's output into the next. Its parameters() collects
# every child's parameters. Give it __len__, __getitem__, and __iter__ so
# you can index into a model and loop over its layers.
#
# Every __init__ must call super().__init__().
#
# Make an abstract Dataset class with abstract __len__ and __getitem__
# methods. Subclass it as a ToyDataset of (x, y) points from y = 3x + 2 plus
# a little noise, with plain floats for x and y.
#
# Practice multiple inheritance with mixins: make a LoggingMixin whose
# forward prints its input and output and calls super().forward(). Then
# make class LoggedLinear(LoggingMixin, Linear). Predict its __mro__ before
# you check it.
#
# Make an abstract Optimizer class with an abstract step method and a
# concrete zero_grad method. Subclass it as SGD, which does
# p.data -= lr * p.grad for every parameter.
#
# Train Linear(1, 1) on the ToyDataset, using MSE loss built from Value
# operations. Check that the loss goes down and that the weight and bias end
# up near 3 and 2. As a bonus, train Sequential([Linear(1, 8), ReLU(),
# Linear(8, 1)]) on y = x**2.
#
# Check your gradients numerically: nudge a value by a tiny eps and compare
# (f(x+eps) - f(x-eps)) / (2*eps) against .grad.
#
# Add type hints everywhere and run mypy. Confirm that Module() raises a
# TypeError.

from abc import ABC, abstractmethod
import numpy as np
import math

# Make a Tensor class that wraps a numpy array and has a .grad attribute.
# Give it the dunder methods __add__, __mul__, __matmul__, __neg__, __sub__,
# and __pow__, so you can write expressions like (x @ W + b) ** 2 directly.
# Each operation should remember its inputs and how to compute their
# gradients. Add a backward() method that walks back through those inputs
# and fills in .grad (basic autograd). Add __repr__ so tensors print
# readably.

import math


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_prev)
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # core ops (each defines its own _backward)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float)), "only number exponents"
        out = Value(self.data ** n, (self,), f"**{n}")

        def _backward():
            self.grad += n * self.data ** (n - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    # derived ops (built from core ops, no new _backward needed)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):   # other + self
        return self + other

    def __rmul__(self, other):   # other * self
        return self * other

    def __rsub__(self, other):   # other - self
        return other + (-self)

    def __rtruediv__(self, other):   # other / self
        return other * self ** -1

    # backprop

    def backward(self):
        order, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                order.append(v)   # added after all its parents

        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


class Tensor:
    ...


    