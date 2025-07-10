"""
funcs.py

Contains functions related to step size change and rms, which is used to
calculate errors.

"""
import numpy as np
from math import sqrt

def rms(x):
	"""
	Evaluates the root mean square of x.
	
	Parameters
	----------
	x : numpy.ndarray
		Array.
	
	Returns
	-------
	float
		RMS of x.
	
	"""
	if x is not np.ndarray: x = np.array(x)

	return np.linalg.norm(x)/sqrt(x.size)

def checkratio(ratio, odeSetup):
	"""
	Limits the step size ratio.
	
	Parameters
	----------
	ratio : float
		Step size ratio. 
	
	Returns
	-------
	float
		Limited ratio.
	
	"""
	return min(ratio, odeSetup.params['maxratio'])

def checkstep(h, odeSetup):
	"""
	Limits the step size.
	
	Parameters
	----------
	h : float
		Step size.
	
	Returns
	-------
	float
		Limited step size.
	
	"""
	intdirection = np.sign(h)
	h = abs(h)
	
	h = min(h, odeSetup.maxstep) # Upper bound
	h = max(h, odeSetup.minstep) # Lower bound
	
	return intdirection*h
	
def init_step(fun, t0, x0, atol, rtol):
	"""
	Calculates the initial step size.
	
	Parameters
	----------
	fun : callable
		ODE function.
	t0 : float
		Initial time
	x0 : float, numpy.ndarray
		Initial condition.
	
	Returns
	-------
	float
		Initial step size.
	
	"""
	tol1 = np.max(rtol)
	
	if tol1==0:
		tol1 = np.max( (atol/x0)[x0!=0] )
	
	tol2 = rtol*np.abs(x0) + atol
	
	dx0 = fun(t0, x0)
	
	return np.sqrt(2*tol1/np.average( (tol1/tol2*dx0)**2 ))

def step_change(Z, ratio):
	"""
	Updates Nordsieck matrix for a step size change.
	
	Parameters
	----------
	Z : numpy.ndarray
		Nordsieck matrix.
	ratio : float
		Step size ratio.
	
	Returns
	-------
	None
	
	"""
	r = np.ones(len(Z))
	for i in range(1,len(r)):
		r[i] = r[i-1]*ratio

	Z*=r[:,np.newaxis]

def colorText(text, kind=None):
	"""
	Writes colored text.
	
	"""
	yellow = '\033[93m'
	red = '\033[91m'
	green = '\033[92m'
	bold = '\033[1m'
	end = '\033[0m'
	
	if kind=='warning':
		return bold + yellow + 'Warning: ' + end + yellow + text + end

	elif kind=='error':
		return bold + red + 'Error: ' + end + red + text + end
	
	elif kind=='success':
		return green + text + end
	
	elif kind=='fail':
		return yellow + text + end
