"""
gear.py

Contains functions that evaluate the next step of the ODE.

"""
from .funcs import *
from .coefficients import factorial, coeffs, mat_S
from .broyden import broyden, functional, jacobian

coeff_dict, B = coeffs() # Generates coefficients

class OdeSetup:
	"""
	Contains variables relevant to the ODE to be integrated.
	
	Attributes
	----------
	odeFun : callable
		Function evaluating the right-hand side of the ODE.
	jacobFun : callable
		Function evaluating the Jacobian of odeFun.
	atol, rtol : float, numpy.ndarray
		Absolute and relative tolerances.
	minstep, maxstep : float
		Minimum and maximum admissible step sizes.
	verbosity : int
		Level of verbosity.
	params : dict
		Dictionary with additional parameters of the integrator.
	
	"""
	
	def __init__(self, odeFun, jacobFun, atol, rtol, minstep, maxstep, verbosity, params):
		self.odeFun = odeFun
		self.jacobFun = jacobFun
		self.atol = atol
		self.rtol = rtol
		self.minstep = minstep
		self.maxstep = maxstep
		self.verbosity = verbosity
		self.params = params
	
class StepConfig:
	"""
	Contains variables relevant to the step that is being evaluated.
	
	Attributes
	----------
	forceOrder : bool
		Variable that defines if order change may be attempted.
	forceStep : bool
		Variable that defines if step size change may be attempted.
	forceMethod : bool
		Variable that defines if integration method change may be attempted.
	
	"""
	
	def __init__(self, forceOrder, forceStep, forceMethod):
		self.forceOrder = forceOrder
		self.forceStep = forceStep
		self.forceMethod = forceMethod
	
class State:
	"""
	State of the step being evaluated.
	
	Attributes
	----------
	t : float
		Time (i.e., independent variable).
	Z : numpy.ndarray
		Nordsieck matrix.
	Zp : numpy.ndarray
		Predictor matrix.
	Z1 : numpy.ndarray
		Trimmed Nordsieck matrix to current order.
	h : float
		Step size. 
	order : int
		Method order.
	method : str
		Integration method.
	invJacob : numpy.ndarray, optional
		Inverse of the Jacobian matrix. Default is None.
	normJacob : float, optional
		Norm of the Jacobian matrix. Default is None.
	d : numpy.ndarray, optional
		Truncation error. Default is numpy.nan.
	tol : numpy.ndarray, optional
		Tolerance. Default is numpy.nan.
	success : bool, optional
		Succes of the step. Default is False.
	
	Methods
	-------
	trim()
		Trims `Z` to current order.	
	step_update(h_new)
		Updates state to a new step size.
	order_update(order)
		Updates state to a new order.
	predictor()
		Calculates `Zp`.
	residual(x, odeSetup)
		Evaluates the residual of the implicit equation.
	inverse_jacobian(odeSetup)
		Calculates the inverse and norm of the Jacobian matrix.
	update_Z(x_new, odeSetup)
		Returns the Nordsieck matrix for a new step.
	scaled_error()
		Evaluates the scaled error.
	solve_step(odeSetup)
		Evaluates the next step from current state.
		
	"""
	
	def __init__(self, t, Z, h, order, method, invJacob=None, normJacob=None, d=np.nan, tol=np.nan, success=False):
		self.t = t
		self.Z = Z
		self.h = h
		self.order = order
		self.method = method
		self.invJacob = invJacob
		self.normJacob = normJacob
		self.d = d
		self.tol = tol
		self.success = success
		
		self.Z1 = self.trim()
		self.Zp = self.predictor()
	
	def trim(self):
		"""
		Trims the Nordsieck matrix to current order.
		
		Returns
		-------
		numpy.ndarray
			Trimmed `Z` matrix.
		
		"""
		return self.Z[0:self.order+1]
	
	def step_update(self, h_new):
		"""
		Updates state to new step size.
		
		Parameters
		----------
		h_new : float
			New step size.
		
		Returns
		-------
		None
		
		"""
		ratio = h_new/self.h # Step size ratio
		self.h = h_new # New step size
		
		step_change(self.Z, ratio)
		step_change(self.Zp, ratio)
		self.Z1 = self.trim()
	
	def order_update(self, order):
		"""
		Updates state to a new order.
		
		Parameters
		----------
		order : int
			New order.
		
		Returns
		-------
		None
		
		"""
		self.order = order
		self.Z1 = self.trim()
		
	def predictor(self):
		"""
		Calculates the predictor `Zp` matrix.
		
		Returns
		-------
		z : numpy.ndarray
			Predictor matrix.
		
		"""
		return B[len(self.Z)]@self.Z
		
#		q = len(self.Z) - 1
#		
#		z = self.Z.copy()
#		
#		for k in range(q):
#			for j in range(q, k, -1):
#				z[j-1] += z[j]

		return z

	def residual(self, x, odeSetup):
		"""
		Evaluates the residual of the implicit equation.
		
		Parameters
		----------
		x : numpy.ndarray
			Solution at current step.
		odeSetup : OdeSetup
			ODE being integrated.
		
		Returns
		-------
		numpy.ndarray
			Residual of the implicit equation.

		"""
		_, K, V, D, _ = coeff_dict[self.method]

		return x - (K[self.order+1]@self.Z1)*V[self.order+1][0] - odeSetup.odeFun(self.t + self.h, x)*self.h*D[self.order+1][0]
		
	def inverse_jacobian(self, odeSetup):
		"""
		Computes the inverse and norm of the Jacobian matrix.
		
		Parameters
		----------
		odeSetup : OdeSetup
			ODE being integrated.
			
		Returns
		-------
		None
		
		"""
		D0 = coeff_dict['BDF'][3][self.order+1][0][0]

		if odeSetup.jacobFun: # Checks if Jacobian was given
			J = np.eye(len(self.Z[0])) - D0*self.h*odeSetup.jacobFun(self.t + self.h, self.Zp[0])
			
		else: # If not, compute Jacobian numerically
			J = jacobian(lambda x: self.residual(x, odeSetup), self.Zp[0], odeSetup)
		
		# Inverse
		self.invJacob = np.linalg.inv(J)
		
		# Norm
		self.normJacob = np.linalg.norm(J/self.tol[:,np.newaxis])
		
	def update_Z(self, x_new, odeSetup):
		"""
		Calculates the Nordsieck `Z` matrix for a new step.
		
		Parameters
		----------
		x_new : numpy.ndarray
			Solution at next step.

		Returns
		-------
		Z_new : numpy.ndarray
			`Z` for next step.

		"""
		# Coefficients	
		S, K, V, D, _ = coeff_dict[self.method]

		Z1 = S[self.order+1]@self.Z1 + K[self.order+1]@self.Z1*V[self.order+1] + D[self.order+1]*self.h*odeSetup.odeFun(self.t + self.h, x_new)
		
		Z_new = self.Z.copy()
		
		Z_new[0:self.order+1] = Z1
		
		return Z_new

	def scaled_error(self):
		"""
		Computes the norm of the scaled truncation error.
		
		Returns:
		float
			Norm of the scaled truncation error
		
		"""
		return rms(self.d/self.tol)

	def solve_step(self, odeSetup):
		"""
		Solves the implicit equation.
		
		Parameters
		----------
		odeSetup : OdeSetup
			ODE being integrated.
		
		Returns
		-------
		State
			State at next step.

		"""
		# Tolerance
		self.tol = odeSetup.atol + odeSetup.rtol*np.abs(self.Z[0])

		# Error constant
		tau = coeff_dict[self.method][-1][self.order]

		# Admissible error for iterative method
		D0 = coeff_dict[self.method][3][self.order+1][0][0]
		broyden_error = D0/( 2*(self.order+2)*tau*factorial(self.order) )
		
		# If BDF, solves with Broyden method
		if self.method=='BDF':
			if self.invJacob is None: self.inverse_jacobian(odeSetup)
			x_new, _, broyden_success = broyden(lambda x: self.residual(x, odeSetup), self.Zp[0], self.invJacob, self.tol, broyden_error, odeSetup)
				
			# Prints message if convergence was not achieved
			if not broyden_success and odeSetup.verbosity>=0:
				print(colorText("maximum number of iterations reached without convergence in Broyden's iteration.", 'warning'))

		# If AM, solves with functional iteration
		elif self.method=='AM':
			x_new, _, functional_success, K_ratio = functional(lambda x: self.residual(x, odeSetup), self.Zp[0], self.tol, broyden_error, odeSetup)

			# Print message if convergence was not achieved
			if not functional_success and odeSetup.verbosity>=0:
				print(colorText("maximum number of iterations reached without convergence in functional iteration.", 'warning'))
				
			self.normJacob = K_ratio/(D0*self.h) # Lower bound of the Jacobian norm

		# Z matrix at next step
		Z_new = self.update_Z(x_new, odeSetup)
		
		# Truncation error
		d = np.abs(Z_new[self.order] - self.Zp[self.order])*factorial(self.order)*tau

		return State(self.t + self.h , Z_new, self.h, self.order, self.method, self.invJacob, self.normJacob, d, self.tol, True)

def gear_update(state, odeSetup, stepConfig):
	"""
	Iterates a single step of the differential equation using Gear's method.
	
	Parameters
	----------
	state : State
		State of current step.
	odeSetup : OdeSetup
		ODE to be integrated.
	stepConfig : StepConfig
		Configurations for the current step.
		
	Returns
	-------
	State
		State of next step.

	"""	
	if state.method=='BDF':
		maxorder = odeSetup.params['BDFmaxorder']
		
	elif state.method=='AM':
		maxorder = odeSetup.params['AMmaxorder']

	order = state.order # Starting order

	# Iteration to reach tolerance in truncation error
	for c in range(odeSetup.params['maxit']):
		# Solution using current order
		state_curr = state.solve_step(odeSetup)
		ratio_curr = 1/( 1.2*(state_curr.scaled_error()**(1/(order+1)) + 1e-6) )
		
		# If order can be changed, tries adjacent orders
		if not stepConfig.forceOrder:
			# Current order + 1
			if order<maxorder:
				state.order_update(order+1) # Changes the order of state
				state_up = state.solve_step(odeSetup)
				ratio_up = 1/( 1.4*(state_up.scaled_error()**(1/(order+2)) + 1e-6) )
				
			else:
				state_up = np.nan
				ratio_up = np.nan

			# Current order - 1
			if order>1:
				state.order_update(order-1) # Changes the order of state
				state_down = state.solve_step(odeSetup) 
				ratio_down = 1/( 1.3*(state_down.scaled_error()**(1/order) + 1e-6) )
				
			else:
				state_down = np.nan
				ratio_down = np.nan

			state.order_update(order) # Changes the order of state back to original

			# Finds best ratio
			best = np.nanargmax([ratio_curr, ratio_up, ratio_down])
			
			state_new = [ state_curr, state_up, state_down ][best]
			ratio =     [ ratio_curr, ratio_up, ratio_down ][best]
			
		# If order change is not allowed, use values from current step
		else:
			state_new = state_curr
			ratio = ratio_curr

		if stepConfig.forceStep: # Does not recalculate the step size if it is forced
			state_new.h = state.h
			
			break

		else: # If step size can be changed, calculates new step size
			# Limits ratio
			ratio = checkratio(ratio, odeSetup)
			
			# Limits step size for next step
			h_new = checkstep(ratio*state.h, odeSetup)

			# Checks if truncation error is within tolerance
			if state_new.scaled_error()<=odeSetup.params['successratio']:
				# Updates state for next step
				state_new.step_update(h_new)
				state_new.success = True
				
				break

			else:
				# Updates state for next attempt
				state.step_update(h_new)
		
	# Sugestion of new method
	# Error constants
	tauA = coeff_dict['AM'][-1][state_new.order]
	tauB = coeff_dict['BDF'][-1][state_new.order]

	if state_new.method=='AM':
		h_A = state_new.h
		h_B = state_new.h*(tauB/tauA)**(-1/(state_new.order+1)) # BDF step size to satisfy tolerance
		
	elif state_new.method=='BDF':
		h_B = state_new.h

		D0A = coeff_dict['AM'][3][state_new.order+1][0][0]
		h_A_it = 1/(2*D0A*state_new.normJacob) # AM step size to ensure functional iteration convergence
		
		h_A_st = state_new.h*(tauA/tauB)**(-1/(state_new.order+1)) # AM step size to satisfy tolerance
		h_A = min(h_A_st, h_A_it) # Most rigorous step size

	# Selects method with largest step size	
	if h_A>h_B: state_new.method = 'AM'
	elif h_B>h_A: state_new.method = 'BDF'
	
	# Force recalculation of Jacobian in next step if method, order or step size is changed
	if state_new.method!=state.method or state_new.order!=state.order or state_new.h!= state.h:
		state_new.invJacob = None
	
	return state_new

