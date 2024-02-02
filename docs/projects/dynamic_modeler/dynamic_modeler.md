# Dynamical Systems and Neural Networks

## Introduction

Systems in nature, finance, and engineering evolve over time in interesting ways. Such systems are called **dynamical systems**. They can also be quite complex and  unpredictable, though in some cases they can be approximated to some sum of predictable patterns. 

Neural networks are able to pick up both simple and complex phenomena while using reasonably generalized training methods, provided they are tuned with some level of expertise.

In this project I build a simulation library for generating signals from dynamical systems, and then predict them using LSTM (Long short term memory) neural networks.

## Dynamical Systems: How do things change?

I'm going to go into a bit more detail on systems changing over time, using some math. If this is confusing, feel free to skip to the section on my applications of neural networks.

Let's say a system has *N* number of variables $$x_1, x_2, ...x_N$$.

We can consider this a dynamical system if the variables' time derivatives depend on the variables in question:

$$\dot{x_1} = f(x_1,x_2, ... x_N)\\
\dot{x_2} = f(x_1,x_2, ... x_N)\\
...\dot{x_N} = f(x_1,x_2, ... x_N)\\$$

**This form of equations is especially convenient for simulations.**

**So how does this work in my simulation engine?**



