"""
This is an example of usage of `geode`. It solves the Robertson's problem, the
most classic example of stiff ODE [1]:

x0' = -0.04 x0 + 1e4 x1*x2                  x0(0) = 1
x1' =  0.04 x0 - 1e4 x1*x2 - 3e7 x1**2      x1(0) = 0
x2' =  3e7 x1**2                            x2(0) = 0

`geode` uses tolerances recommended by [2] and returns solutions at t = 4e-1,
4e0, ... 4e10, which can be compared to the solution obtained in [2].

[1] H. Robertson. The solution of a set of reaction rate equations. 1966.
[2] K. Radhakrishnan, A. Hindmarsh. Description and use of LSODE... 1993.

"""
import numpy as np
from geode import *

def odefun(t, x):
	"""
	Right-hand side of the Robertson's set of equations.
	
	"""
	dx0 = -.04*x[0] + 1e4*x[1]*x[2]
	dx1 =  .04*x[0] - 1e4*x[1]*x[2] - 3e7*x[1]**2
	dx2 =  3e7*x[1]**2

	return np.array([dx0, dx1, dx2])

def jacobfun(t, x):
	"""
	Jacobian of `odefun`.
	
	"""
	J = np.zeros([3,3])
	
	J[0,0] = -.04
	J[0,1] = 1e4*x[2]
	J[0,2] = 1e4*x[1]
	
	J[1,0] = .04
	J[1,1] = -1e4*x[2] - 2*3e7*x[1]
	J[1,2] = -1e4*x[1]
	
	J[2,0] = 0
	J[2,1] = 2*3e7*x[1]
	J[2,2] = 0
	
	return J

# Initial integration time
t0 = 0

# Output stations
t1 = np.geomspace(4e-1, 4e10, 12)

# Initial conditions
x0 = np.array([1, 0, 0])

# Absolute tolerance
atol = np.array([1e-6, 1e-8, 1e-6])

# Relative tolerance
rtol = 1e-4

# Calls geode
t, x = geode(odefun, x0, t0, t1, atol=atol, rtol=rtol, jacobfun=jacobfun)

# Prints solution
for i in range(len(t1)):
	print(f'{t[i]:1.0E} & {x[i,0]:1.6E} & {x[i,1]:1.6E} & {x[i,2]:1.6E} \\\\')
