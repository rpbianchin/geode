"""
broyden.py

jacobian approximates the Jacobian matrix by finite differences, and Broyden
solves a system of equations. broyden is called by gear.solve_step.

"""
import numpy as np
from .funcs import rms
from .coefficients import factorial

def jacobian(fun, x, odeSetup):
	"""
	Approximates the Jacobian of a function by finite differences.
	
	Parameters
	----------
	fun : callable
		Function.
	x : numpy.ndarray
		Argument of `fun` for the evaluation of the Jacobian.
	
	Returns
	-------
	numpy.ndarray
		Jacobian of `fun` at `x`.
	
	"""
	numvars = len(x) # Number of variables
	
	dx = odeSetup.params['reljacob']*np.abs(x) + odeSetup.params['absjacob'] # Step for finite differences
	
	J = np.zeros([numvars,numvars])
	
	f0 = fun(x) # Function at given point
	
	for i in range(numvars): # For each variable in x
		# Step size array
		Dx = np.zeros(numvars)
		Dx[i] = dx[i]

		f = fun(x + Dx)
		
		if odeSetup.params['jacoborder']==2:
			f2 = fun(x + 2*Dx) # Extra point if using order 2

		for j in range(numvars): # For each function in f
			if odeSetup.params['jacoborder']==1: J[j,i] = (f[j] - f0[j])/Dx[i]
			elif odeSetup.params['jacoborder']==2: J[j,i] = (-1.5*f0[j] + 2*f[j] - .5*f2[j])/Dx[i]
			
	return J

def broyden(fun, x0, Ijacob, tol, error, odeSetup):
	"""
	Finds the zero of a function using the good Broyden's method.
	
	Parameters
	----------
	fun : callable
		Function.
	x0 : numpy.ndarray
		Initial guess.
	Ijacob : np.ndarray
		Inverse of the Jacobian of `fun` at `x0`.
	tol : np.ndarray
		Tolerance array for current step.
	error : np.ndarray
		Admissible error for current step.
	
	Returns
	-------
	x : numpy.ndarray
		Zero of the function.
	f : numpy.ndarray
		Residual at last iteration.
	success : bool
		Success of iteration.	
	
	"""
	IJ = Ijacob # Inverse Jacobian
	f0 = fun(x0)[:, np.newaxis] # Initial value of the function
	x0 = x0[:, np.newaxis] # Reshaped initial guess

	success = True

	count = 0

	while True:
		x = x0 - IJ@f0

		f = fun(x[:,0])[:, np.newaxis]

		Df = f - f0
		Dx = x - x0

		IJDf = IJ@Df

		if odeSetup.params['broydentype']=='good':	
			IJ += (Dx - IJDf)@Dx.T@IJ/(Dx.T@IJDf)
			
		elif odeSetup.params['broydentype']=='bad':		
			IJ += (Dx - IJDf)@Df.T/(Df@Df.T)
		
		if rms((Dx - Df)/tol)<=error: # Convergence criterion
			break
		
		if count>odeSetup.params['maxbroydenit']: # Maximum number of iterations
			success = False
			break

		else:
			# Updates x0 and f0
			x0 = x
			f0 = f
			
		count += 1
		
	return x[:,0], f[:,0], success

def functional(fun, x0, tol, error, odeSetup):
	"""
	Finds the zero of a function using functional iteration.
	
	Parameters
	----------
	fun : callable
		Function.
	x0 : numpy.ndarray
		Initial guess.
	tol : np.ndarray
		Tolerance array for current step.
	error : np.ndarray
		Admissible error for current step.
	
	Returns
	-------
	x : numpy.ndarray
		Zero of the function.
	f : numpy.ndarray
		Residual at last iteration.
	success : bool
		Success of iteration.
	ratio : float
		Ratio of successive differences for estimation of rate of convergence.
	
	"""
	count = 0

	success = True

	xa = [0, x0, 0] # Array with previous values

	f0 = fun(x0)
	
	# First step
	x = x0 - f0
	
	xa = [x, x0, 0]
	
	x0 = x
	
	ratio = -np.inf
	
	while True:
		x = x0 - f0 # Calculates new x

		# Shifts xa
		xa[1:2] = xa[0:1]
		xa[0] = x

		# Calculates ratio
		new_ratio = rms( (xa[0] - xa[1])/tol ) / rms( (xa[1] - xa[2])/tol )
		if new_ratio>ratio: ratio = new_ratio

		Dx = -f0

		f = fun(x)
		
		Df = f - f0
		
		if rms((Dx-Df)/tol)<=error: # Convergence criterion
			break

		if count>odeSetup.params['maxbroydenit']: # Maximum number of iterations
			success = False
			break

		else:
			# Updates x0 and f0
			x0 = x
			f0 = f
			
		count += 1
		
	return x, f, success, ratio
