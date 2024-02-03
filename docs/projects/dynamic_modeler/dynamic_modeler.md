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

**My simulation engine can generate signals for any system that can be rewritten in the above form.**

**Let's look at an example**

Behold, the damped driven oscillator:

Consider an object with mass m attached to a spring with "spring constant" k. In this example, there is also a driving force $$F(t)$$ pushing and pulling on the spring.

You will find this situation often written in the following standard form:

$$m\ddot{x} + b\dot{x} + kx = F(t)$$

However, we need time derivatives as explicit functions. So we rewrite the above as:

$$\ddot{x} = \dfrac{1}{m}(F(t) - b\dot{x} - kx)$$

If we define the mass's position $$x_1 \coloneqq x$$ and the mass's velocity $$x_2 \coloneqq \dot{x}$$, then we essentially have the variables' time derivatives expressed as functions in the previously mentioned general form:

$$
\dot{x_1} = \dot{x_2}\\
\dot{x_2} = \dfrac{1}{m}(F(t) - bx_2 - kx_1)
$$

**How does my simulation engine put these equations into practice?**

We first inform the engine initial conditions at time $t=0$, the 

## Engine Code Structure