# Dynamical Systems and Neural Networks

## Introduction

Systems in nature, finance, and engineering evolve over time in interesting ways. Such systems are called **dynamical systems**. They can also be quite complex and  unpredictable, though in some cases they can be approximated to some sum of predictable patterns. 

Neural networks are able to pick up both simple and complex phenomena while using reasonably generalized training methods, provided they are tuned with some level of expertise.

In this project I build a simulation library for generating signals from dynamical systems, and then predict them using LSTM (Long short term memory) neural networks.

## Dynamical Systems: How do things change?

I'm going to go into a bit more detail on systems changing over time, using some math. If this is confusing, feel free to skip to the section on my applications of neural networks.

Let's say a system has *N* number of variables $x_1, x_2, ...x_N$.

We can consider this a dynamical system if the variables' time derivatives depend on the variables in question:

$$\dot{x}_1 = f(x_1,x_2, ... x_N)\\
\dot{x}_2 = f(x_1,x_2, ... x_N)\\
...\dot{x}_N = f(x_1,x_2, ... x_N)\\$$

**My simulation engine can generate signals for any system that can be rewritten in the above form.**

**Let's look at an example**

Behold, the damped driven oscillator:

Consider an object with mass m attached to a spring with "spring constant" k. There is also a friction constant b which damps the oscillation of the spring. Finally, there is also a driving force $F(t)$ pushing and pulling on the spring.

Let's say someone pulls on the mass away from the spring's equilibrium and let's go, this would lead to a harmonic oscillation, in addition to the driving force.

You will find this situation often written in the following standard form:

$$m\ddot{x} + b\dot{x} + kx = F(t)$$

However, we need time derivatives as explicit functions. So we rewrite the above as:

$$\ddot{x} = \dfrac{1}{m}(F(t) - b\dot{x} - kx)$$

If we define the mass's position $x_1 \coloneqq x$ and the mass's velocity $x_2 \coloneqq \dot{x}$, then we essentially have the variables' time derivatives expressed as functions in the previously mentioned general form:

$$
\dot{x}_1 = x_2\\
\dot{x}_2 = \dfrac{1}{m}(F(t) - bx_2 - kx_1)
$$

**How does my simulation engine put these equations into practice?**

We first inform the engine initial conditions at time $t=0$. This includes the mass's initial velocity $\dot{x}_{t=0}$ and initial position $x_{t=0}$. In addition to knowing the driving force at all times, this allows us to calculate the initial acceleration:

$\ddot{x}_{t=0} = \dfrac{1}{m}(F(t=0) - b\dot{x}_{t=0} - kx_{t=0})$

Then, given a discrete time step size $dt$, use $\ddot{x}_0$ to calculate the velocity at the next time step:

$$\dot{x}_{t=dt} = \ddot{x}_{t=0}dt$$

Repeat this over and over to generate a signal, iterating through values of $t=m\mathrm{d}t$:

$$\dot{x}_{t=(m+1)dt} = \ddot{x}_{t=mdt}$$

**What does the simulation end up looking like?**

Let's say I've got a driving force defined as a sine wave. Perhaps it's an elf meticulously pushing and pulling on the spring:

$$F(t) = \sin(2\pi t)$$

Furthermore I set the initial conditions and constant values as follows:

* mass $m = 1$
* friction constant $b = 0.1$
* spring constant $k = 0.1$
* initial position $x_{t=0} = 0$
* initial velocity $\dot{x}_{t=dt} = 0.5$

Plugging this into my engine we get the following signal:

![damped_oscillator](damped_oscillator.png)

## How to Use the Engine

Now that we've looked at an example, you may be wondering, **how do I use the engine to simulate my own nonlinear equation?**

The engine revolves around one Python function that I call "iterate". All you need to do is pass the following items (i.e. arguments):

* The initial state vector (1 dimension), which contains the variables' values at initial time $t_0$. In the case of the damped oscillator, it would be a list of initial position $x_{t=0}$ and initial velocity $\dot{x}_{t=dt}$.
* The value of the discrete time step $dt$, i.e. the time of each frame.
* Number of time steps or frames $N$ for which the simulation takes place
* A list of functions that take calculate the state vector $x$'s time derivatives for each variable. For the damped oscillator this would be calculating $\dot{x}$ and $\ddot{x}$.
* A dictionary of custom arguments used (if needed) for the functions

**How did I use this for my damped oscillator?**

I define a function that takes intiial values and constants specific to a damped oscillator and feeds it into the simulation engine.

You will see that functions similar to the above form are created for velocity and acceleration and fed into the the x_iterate function.

```
def x_driven(x_t0, dt, N, m, b, k, u1, args):
    """
    Description:
        Simulate damped driven oscillator, with driving force u_1(t)
    x_t0: 1 dimensional array (torch tensor), shape n
            contains n values at time t = 0
    dt: scalar value
            timestep quantity
    N: scalar value
        number of time steps to be iterated through
    m: scalar value
        mass constant
    b: scalar value
        friction constant
    k: scalar value
        spring constant
    u: 1 dimensional list, shape n
        contains n functions to be used on x; input functions in a system
    args: dictionary
        arguments for function u
    """
    #Prepare system functions
    #Velocity
    def xdot(x, t, args):
        return x[1]
    #Acceleration
    def xdotdot(x, t, args):
        return (u1(t, args) - b*x[1] - k*x[0])/m
    
    #Iterate through the time steps to calculate the variables using the system functions
    x_full = x_iterate(x_t0=x_t0, dt=dt, N=N, f=[xdot, xdotdot], args=args)
```

After importing the library, it's as simple as using the damped oscillator function. In this case I use a custom function called harmonics which essentially spits out a sum of sine waves (or in this case 1). You can try putting in multiple sine waves of multiple amplitudes and frequencies for fun:

```
#Import source code
from src import dynamicmodel as dm

#Numeric/Computational libraries
import numpy as np

#Initialize parameters
dt =.02
N = 5000

#Initialize u at t=0
x_t0 = np.array([0,0.05])

x_array_dampeddriven = dm.x_driven(x_t0=x_t0, dt=dt, N=N, m=1, b=0.1, k=1, u1=dm.harmonics, args={'n_list':[1], 'a_list':[1]})
```
**I also used my engine to simulate an RLC circuit, check out the repo to see more!**
In this RLC circuit simulation we have:
* The capacitor voltage $V_C$ with initial value at $t=0$
* The inductor current $I_L$ with initial value at $t=0$
* The capacitor's capacitance $C$ (defined to be 1 in this simulation)
* The resistor's resistance $R$ (defined to be 1 in this simulation)
* The voltage $V(t)$ from the battery in the circuit, ($\sin(2\pi t)$ in this simulation)

$$
\dot{V}_C = \dfrac{I_L}{C}
$$
$$
\dot{I}_L = \dfrac{1}{L}(-V_C - RI_L + V(t))
$$
![rlc](rlc.png)

## Neural Network Predictions
