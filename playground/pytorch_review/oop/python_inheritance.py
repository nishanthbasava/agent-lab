# Reviewing Python OOP
# https://www.youtube.com/watch?v=OemVdsibSFQ

# ABC = Abstract Base Class
from abc import ABC, abstractmethod #abstraction, setting rules.

class Staff(ABC):
    role_call = []

    def __init__(self, name, shift):
        self.name = name
        self.shift = shift
        self.role_call.append(self.name)

    @abstractmethod # this just means that subclasses MUST override it
    def experience(self):
        pass #no implementation, just says what MUST exist in subclasses

    @abstractmethod
    def hello(self):
        return f"{self.name} says hello!" # here, has an implementation, so subclasses can reference it with super()

    def start_work(self):
        print(f"{self.name} starts working")

#INHERITANCE

class Waiter(Staff):
    def __init__(self, name, shift, chef, exp :int = 1):
        super().__init__(name, shift)
        self.chef = chef
        self.exp = exp

    def hello(self):
        print(super().hello())

    def take_order(self, dish):
        print(f"{self.name} takes order: {dish}")
        self.chef.cook_food(dish)

    def order_of_fish(self):
        self.take_order("fish")

    def experience(self, exp=1):
        self.exp=exp
        if self.exp == 0:
            print("Does not have any experience.")
        else: 
            print(f"{exp} years serving customers")
    
class Chef(Staff):
    def cook_food(self, dish):
        print(f"Chef {self.name} cooks {dish}")

    def work(self):
        self.cook_food()

    def hello(self, message="welcome"):
        print(super().hello() + f" and also wanted to say {message}")

    def experience(self):
        print("3 years serving customers")
    
Raj = Chef("Raj", "night")
Amit = Waiter("Amit", "morning", Raj, 3)

Amit.take_order("biryani")
Amit.order_of_fish()
Raj.hello()
Raj.experience()
Amit.experience(0)

print(Staff.role_call)

# POLYMORPHISM
# same instruction, many forms of behavior

# ABC (Abstract Base Class) 
# this is Python's built in way to define abstract classes
# for a class to be abstract, it must have at least one abstract method


# Composition
# Building something by combining smaller parts

# inheritance: IS-A
# composition: HAS-A

# Encapsulation: an object controls its own data. We don't change data from outside; we ask the object to change it using a method
# State is just the objects current data.