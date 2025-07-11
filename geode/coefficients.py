"""
coefficients.py

Functions for the generation of Gear's method coefficients. Function coeffs is
called by globars.setup, and factorial is called by gear.solve_step.

"""
import numpy as np

def factorial(n):
	"""
	Look-up table for the factorial.
	
	Parameters
	----------
	n : int
		Argument.

	Returns
	-------
	int
		Factorial of `n`.
		
	"""
	return [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200][n]

def binomial(n,k):
	"""
	Calculates the binomial coefficient.
	
	Parameters
	----------
	n, k : int
		Arguments.
	
	Returns
	-------
	float
		Binomial coefficient.
		
	"""
	if n>=k and k>=0: return factorial(n)/( factorial(k)*factorial(n-k) )
	else: return 0

def coeffs():
	"""
	Generates Gear's method coefficients.

	Parameters
	----------
	None

	Returns
	-------
	r, a, p, B : list
		Lists with the coefficients for each order.

	"""
	N = 15 # Generates coefficients up to order 15

	# Creates empty lists
	S, K, V, D = [[]]*N, [[]]*N, [[]]*N, [[]]*N
	Sh, Kh, Vh, Dh = [[]]*N, [[]]*N, [[]]*N, [[]]*N

	C = [[]]*N
	tau = [[]]*N
	tauh = [[]]*N
	
	B = [[]]*N
	
	for n in range(N-1):
		C[n] = inv_vandermonde(n+1) # BDF coefficients
		E = adams_coeff(n-1) # AM coefficients
		
		B[n] = mat_B(n)
		
		if n>0:
			# History to Nordsieck matrices
			Q = C[n-1]
			iQ = vandermonde(n)
		
			# BDF matrices
			A = mat_A(n) # Lower shift matrix
						
			P = C[n][1,:][1::]

			S[n] = Q @ A @ iQ

			K[n] = -P @ iQ
			
			V[n] = np.array([ Q[:,0] ]).T/C[n][1,0]
			
			D[n] = V[n]
			
			tau[n] = 1/(n+1) # Error constant
		
			# AM matrices
			Ah = A.copy()
			if n>1: Ah[1] = 0
			
			F = np.zeros([n,1])
			F[0] = E[0]
			if n>1: F[1] = 1
			
			Qh = np.zeros([n,n])
			Qh[1::,1::] = C[n-2]
			Qh[0,0] = 1
			iQh = np.linalg.inv(Qh)

			Ph = E.copy()
			Ph[0] = 1

			Sh[n] = Qh @ Ah @ iQh
			
			Kh[n] = Ph@iQh
			
			Vh[n] = np.zeros([n,1])
			Vh[n][0] = 1
			
			Dh[n] = Qh@F
			
			tauh[n] = np.abs(1/factorial(n+1) - 1/factorial(n)*np.sum( [(i-1)**n*E[i] for i in range(0,n)] )) # Error constant

	coeff_dict = { 'AM' : [Sh, Kh, Vh, Dh, tauh], 'BDF' : [S, K, V, D, tau] }

	return coeff_dict, B
			
def vandermonde(N):
	"""
	Generates the Vandermonde matrix for a given size N.

	Parameters
	----------
	N : int
		Matrix size.

	Returns
	-------
	numpy.ndarray
		Vandermonde matrix.

	"""
	V = np.zeros([N,N])
	
	for i in range(N):
		for j in range(N):
			V[i,j] = (-i)**j		
	
	return V

def inv_vandermonde(N):
	"""
	Generates the inverse Vandermonde matrix for a given size N.

	Parameters
	----------
	N : int
		Matrix size.

	Returns
	-------
	numpy.ndarray
		Inverse Vandermonde matrix.

	"""
	L = np.zeros([N,N])
	U = np.eye(N)
	
	L[0,0] = 1
	
	for i in range(N):
		for j in range(N):
			if i>=j and i>0:
				L[i,j] = (-1)**j/(factorial(j)*factorial(i-j)) # Inverse of the lower factor of the Vandermonde matrix
				
			if j>0:
				U[i,j] = U[i-1,j-1] + (j-1)*U[i,j-1] # Inverse of the upper factor of the Vandermonde matrix
	
	return U @ L

def mat_A(N):
	"""
	Generates the lower shift matrix `A` for a given size N.

	Parameters
	----------
	N : int
		Matrix size.

	Returns
	-------
	numpy.ndarray
		`A` matrix.

	"""
	A = np.zeros([N, N])

	for i in range(N):
		for j in range(N):
			if j==i-1:
				A[i,j] = 1

	return A

def mat_B(N):
	"""
	Generates the binomial matrix `B` for a given size `N`.

	Parameters
	----------
	N : int
		Matrix size.

	Returns
	-------
	numpy.ndarray
		`B` matrix.

	"""
	B = np.zeros([N, N])
	
	for i in range(N):
		for j in range(N):
			B[i,j] = binomial(j,j-i)
	
	return B
	
def mat_S(N, ratio):
	"""
	Generates the matrix `S` for a given size `N`.

	Parameters
	----------
	N : int
		Matrix size.

	Returns
	-------
	numpy.ndarray
		`S` matrix.

	"""
	S = np.eye(N)
	
	r = 1
	
	for i in range(N):
		S[i,i]*=r
		r*=ratio

	return S

def polymult(a,b):
	"""
	Multiplies a polynomial `a` in the form a0*x^n + a1*x^(n-1)... to a binomial
	`b` in the form b0*x + b1.
	
	Parameters:
	-----------
	a : numpy.ndarray
		Polynomial.
	
	b : numpy.ndarray
		Binomial.
	
	Returns:
	--------
	numpy.ndarray
		Multiplication of the polynomials.
	
	"""
	c = np.zeros(len(a)+1)
	
	c[0:-1] = a*b[0]
	c[1::] += a*b[1]
	
	return c

def polyint(a):
	"""
	Computes the antiderivative of a polynomial `a` in the form a0*x^n + a1*x^(n-1)....
	
	Parameters:
	-----------
	a : numpy.ndarray
		Polynomial.
	
	Returns:
	--------
	numpy.ndarray
		Antiderivative of the polynomial.
	"""	
	b = np.zeros(len(a)+1)
	
	for i in range(0,len(a)):
		b[i] = a[i]/(len(a) - i)
	
	return b

def polyval(a,x0):
	"""
	Evaluates a polynomial in the form a0*x^n + a1*x^(n-1)... at x=x0 using
	Horner's scheme.
	
	Parameters:
	-----------
	a : numpy.ndarray
		Polynomial.
	
	x0 : numpy.ndarray
		Point to evaluate the polynomial.
	
	Returns:
	--------
	float
		Evaluated polynomial.
	"""
	p = a[0]
	
	for c in a[1::]:
		p = x0*p + c
	
	return p

def adams_coeff(N):
	"""
	Computes the coefficients of Adams-Moulton method of order `N`.
	
	Parameters:
	-----------
	N : int
		Order.
	
	Returns:
	--------
	numpy.ndarray
		Coefficients.
	"""
	E = np.zeros(N+1)

	for i in range(N+1):
		if i==0: p = np.array([0,1]) # p = 1
		else: p = np.array([1,-1]) # p = (u - 1)
		
		for m in range(1,N+1):
			if m!=i:
				p = polymult(p, np.array([1,m-1])) # p = p*(u + m - 1)
			
		E[i] = (-1)**i*polyval(polyint(p),1)/(factorial(i)*factorial(N-i))
	
	return E
			
