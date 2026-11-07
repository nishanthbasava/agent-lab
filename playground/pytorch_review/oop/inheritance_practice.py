# SHAPE LIBRARY PRACTICE PROJECT
#
# Make an abstract base class called Shape that inherits from ABC. Give it
# two abstract methods, area and perimeter, whose bodies are just "...".
# Give it a third abstract method called describe that does have a body and
# returns the string "Shape". Also give Shape a normal method called scale
# that takes a factor k and that every subclass inherits without overriding.
#
# Make two subclasses, Rectangle and Circle, that implement area, perimeter,
# and describe. Their describe methods should call super().describe() and
# add their own text to what it returns.
#
# Add dunder methods to Shape so all shapes behave like built-in values.
# __repr__ should print something readable, __eq__ and __lt__ should compare
# shapes by area, __add__ should return the sum of two shapes' areas, and
# __mul__ should return a scaled copy of the shape.
#
# Make Square inherit from Rectangle. Its __init__ should take one side
# length and call super().__init__(side, side) instead of repeating
# Rectangle's setup.
#
# Practice multiple inheritance: make two mixin classes, ColorMixin and
# LabelMixin, each with a describe method that calls super().describe() and
# adds a color or a label. Then make FancySquare(ColorMixin, LabelMixin,
# Square). Before running anything, write down what you think
# FancySquare.__mro__ is and what describe() will print, then check both.
#
# Make a ShapeCollection class that holds a list of shapes. Give it __len__,
# __getitem__, __iter__, and __contains__ so it works with len(), indexing,
# for loops, and the "in" keyword. Check that sorted(collection) orders the
# shapes by area.
#
# Add type hints to every function and method, then run mypy and fix
# whatever it reports.
#
# To finish, confirm that Shape() raises a TypeError and that a subclass
# missing perimeter can't be instantiated.

