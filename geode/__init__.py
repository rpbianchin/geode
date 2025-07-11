"""
__init__.py

"""
from .gear import *
from .funcs import colorText

default_params = {
	'BDFmaxorder':     5,      # Maximum order for BDF method
	'AMmaxorder':      12,     # Maximum order for AM method
	'maxratio':        2,      # Maximum ratio of step size increase
	'successratio':    1,      # Convergence criterion for ratio
	'maxit':           10,     # Maximum attempts at convergence per step
	'jacoborder':      2,      # Order of finite differences approximation for Jacobian
	'absjacob':        1e-10,  # Absolute factor for Jacobian step size
	'reljacob':        1e-6,   # Relative factor for Jacobian step size
	'maxbroydenit':    25,     # Maximum number of iterations in Broyden method
	'broydentype':     'good', # Type of Broyden method
	'orderchange':     5,      # Frequency of order change
	'stepchange':      10,     # Frequency of step size change
}

def geode(odefun, x0, t0, t1, atol=0, rtol=1e-6, jacobfun=None, method='hybrid',
          minstep=0, maxstep=np.inf, initstep=None, starting_method='am',
          full_output=False, verbosity=0, params={}):
	"""
	Integrates a differential equation using Gear's method.

	Parameters
	----------
	odefun : callable
		Function evaluating the right-hand side of the ODE.
	x0 : float or numpy.ndarray
		Initial condition.
	t0 : float
		Start time.
	t1 : float or iterable.
		If the argument is a float, it corresponds to the time of the end of the
		integration interval. If the argument is an iterable, it corresponds to
		the stations at which the solution will be calculated for the output. In
		the latter case, the list or array must be ordered. 
	atol : float or numpy.ndarray, optional
		Absolute tolerance for truncation error. If it is a numpy.ndarray, it
		must have the same shape as `x0`. Default is 0.
	rtol : float or numpy.ndarray, optional
		Relative tolerance for truncation error. If it is a numpy.ndarray, it
		must have the same shape as `x0`. Default is 1e-6.
	jacobfun : callable, optional
		Jacobian of the differential equation. If none is given, the Jacobian 
		is calculated numerically. Default is None.
	method : str, optional
		Method to be used by the integrator. Options are 'am' for the Adams-
		Moulton method, 'bdf' for the Backwards Differentiation Formula and
		'hybrid', in which case the integrator will switch methods
		automatically. Default is 'hybrid'.
	minstep, maxstep : float, optional
		Minimum and maximum step sizes allowed. Default is `minstep`=0 and
		`maxstep`=numpy.nan.
	initstep : float, optional
		Initial step size. If none is given, it is calculated to satisfy
		tolerances. Default is None.
	starting_method : str, optional
		Integration method to be used in the first step. Default is 'am'.
	full_output : bool, optional
		If True, returns optional outputs. This argument cannot be used if `t1`
		is an iterable. Default is False.
	verbosity : int, optional
		Level of verbosity. If -1, nothing will be printed. If 0, warnings and
		errors will be shown. If larger than 0, information about the integration
		will be shown every `verbosity` steps. Default is 0.
	params : dict, optional
		Dictionary with additional parameters for the integrator. If none is
		given, default parameters are used. Default is {}.

	Returns
	-------
	t : numpy.ndarray
		Time (i.e., independent variable) array.
	x : numpy.ndarray
		Solution array.
	d : numpy.ndarray
		Truncation error in each step. Optional.
	tol : numpy.ndarray
		Tolerance in each step. Optional.
	ord : numpy.ndarray
		Order used in each step. Optional.
	method : numpy.ndarray
		Method used in each step. Optional.
	success : numpy.ndarray
		Success status of each step. Optional.
		
	"""
	## Sets up parameter dictionary
	aux_params = default_params.copy()
	params.update(aux_params)
	
	if type(x0)!=np.ndarray: # If x0 is a float, converts to an array
		x0 = np.array([x0])
	
	## Checks if output stations are used
	if type(t1)==list: # If t1 is a list, transforms into an array
		t1 = np.array(t1)
	
	if type(t1)==np.ndarray: # If t1 is an array, output stations are used
		out_stations = True
		last_time = t1[-1] # Last integration step
		
		if full_output: # Raise error if full output is expected
			raise ValueError("Full output is not available if argument 't1' is not a float.")
	else:
		out_stations = False
		last_time = t1
	
	neq = len(x0) # Number of equations
	
	## Defines direction of integration
	intdirection = np.sign(last_time - t0)
	
	if np.isinf(maxstep): maxstep = intdirection*np.inf

	## Validates atol and rtol
	if type(atol)==np.ndarray:
		if len(atol)!=1 and len(atol)!=neq:
			raise ValueError("Argument 'atol' must be a float, an array with one element or an array with the same shape as x0.")

	if type(rtol)==np.ndarray:
		if len(rtol)!=1 and len(rtol)!=neq:
			raise ValueError("Argument 'rtol' must be a float, an array with one element or an array with the same shape as x0.")

	## Validates method
	if method not in ['AM', 'am', 'BDF', 'bdf', 'hybrid']:
		print(colorText("invalid argument 'method' %s. Defaulting to 'hybrid'."%method, 'warning'))
		method = 'hybrid'

	if method=='hybrid':
		hybrid = True
		method = starting_method # Initial method if hybrid
	else:
		hybrid = False
		
	if method=='am': method = 'AM'
	elif method=='bdf': method = 'BDF'
		
	## Initializes arrays
	if out_stations:
		tout = t1.copy() # Output times are given by t1
		x = np.zeros([len(tout), len(x0)]) # x shape is known
		next_station = 0 # Next output station to look for
	else:
		x = np.array([x0]) # Dependent variable
	
	t = np.array([t0]) # Independent variable (complete time array)

	## Initial step size
	if initstep is None:
		h = intdirection*init_step(odefun, t0, x0, atol, rtol)
	else:
		h = initstep

	if full_output:
		d_arr = np.zeros(x.shape) # Truncation error
		tol_arr = np.zeros(x.shape) # Tolerance
		met_arr = np.array([method]) # Method

	stp_arr = np.array([h]) # Step size
	ord_arr = np.array([1]) # Method order
	suc_arr = np.array([True]) # Success of step

	# Length of the Nordsieck matrix
	if hybrid:
		lenZ = max(params['BDFmaxorder'], params['AMmaxorder']) + 1
	elif method=='AM':
		lenZ = params['AMmaxorder'] + 1
	elif method=='BDF':
		lenZ = params['BDFmaxorder'] + 1

	# Nordsieck matrix
	Z = np.zeros([ lenZ, len(x0)])
	Z[0] = x0
	Z[1] = h*odefun(t0,x0)
	
	count = 0
	
	forceorder = True # Variable to define if order change is allowed
	forcestep = False # Variable to define if step size change is allowed
	forcemethod = False # Variable to define if method change is allowed
	
	last_order_attempt = 0
	last_step_attempt = 0
	
	odeSetup = OdeSetup(odefun, jacobfun, atol, rtol, minstep, maxstep, verbosity, params)
	stepConfig = StepConfig(forceorder, forcestep, forcemethod)
	state = State(t0, Z, h, 1, method)

	# Main loop
	while intdirection*t[-1]<intdirection*last_time:
		# Calculates next step
		try:
			# Main calculation
			state_new = gear_update(state, odeSetup, stepConfig)

			# Error in step
			if np.nan in state_new.Z or np.isnan(state_new.h):
				raise ValueError

		# Handles error in step
		except ValueError:
			if verbosity>=0:
				print(colorText('integration was halted because invalid values were found in the solution.', 'error'))
			break
		
		# If output stations are used
		if out_stations:
			while True: # Look for all stations between last step and next
				if intdirection*t_new>intdirection*tout[next_station]:
					r = (tout[next_station] - t_new)/state_new.h # Ratio
					
					# Computes solution using Taylor series
					x[next_station] = state_new.Z_new[-1]
					for z in state_new.Z_new[-2::-1]:
						x[next_station] = x[next_station]*r + z

					if next_station<len(tout)-1: # Goes to next station
						next_station += 1
					else: # Last station was already calculated
						break
				
				else: # No more stations in current step
					break

		# If output stations are not used
		else:
			# Checks if integration limit was crossed and recalculates
			if intdirection*state_new.t>intdirection*last_time:
				state.step_update(last_time - t[-1]) # Final step to reach integration limit
				
				stepConfig.forceOrder = True
				stepConfig.forceStep = True
				
				state_new = gear_update(state, odeSetup, stepConfig)

			# Updates solution
			x = np.vstack([x, np.array(state_new.Z[0])])
			
		# Updates time
		t = np.append(t, state_new.t)
		
		# Updates error and tolerance and method arrays
		if full_output:
			d_arr = np.vstack((d_arr, state_new.d))
			tol_arr = np.vstack((tol_arr, state_new.tol))
			met_arr = np.append(met_arr, state_new.method)
		
		# Updates order, success and step size arrays
		ord_arr = np.append(ord_arr, state_new.order)
		suc_arr = np.append(suc_arr, state_new.success)
		stp_arr = np.append(stp_arr, state_new.h)
		
		# Allows order change after `orderchange` consecutive steps with no order change
		if np.all(ord_arr[-1:-params['orderchange']:-1]==ord_arr[-1]) and count-last_order_attempt>=params['orderchange']:
			stepConfig.forceOrder = False
			last_order_attempt = count
		else:
			stepConfig.forceOrder = True

		# Allows step size change after `stepchange` consecutive successful steps
		if np.all(suc_arr[-1:-params['stepchange']:-1]):
			stepConfig.forceStep = False
			last_step_attempt = count
		else:
			stepConfig.forceStep = True
		
		count += 1
		
		# Prints status
		if verbosity>0 and (count%verbosity==0 or t[-1]>=t1):
			print('Step {:5d} | h = {:.2e} | t = {:.2e} | order = {:d} | error = {:.2e} | {:3s} | {:s}'.format(count, state_new.h, t[-1], state_new.order, rms(state_new.d/state_new.tol), state.method, [colorText('Fail','fail'), colorText('Success', 'success')][state_new.success]))

		# Updates state for next step
		state = state_new

	if out_stations:
		t = tout		

	if full_output: # Returns full output
		return t, x, d_arr, tol_arr, ord_arr, met_arr, suc_arr
	
	else:
		return t, x
