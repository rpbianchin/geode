# geode

**GEar solver for ODEs** is an integrator for stiff ordinary differential equations. It uses Gear's method with Adams-Moulton method for non-stiff problems and Backwars Differentiation Formula for stiff problems. `geode` is heavily inspired on LSODA by A. Hindmarsh and L. Petzold. To maintain accuracy without compromising performance, `geode` changes step size, method and order automatically.

## Usage

The basic syntax for `geode` is
```python
t, x = geode(odefun, x0, t0, t1)
```
where `odefun` is a function evaluating the right-hand side of the ODE; `x0` is the initial condition; and `t0` and `t1` are the bounds of the integration interval. The outputs `t` and `x` are the values of the independent variable and solution at each integration step.

Instead of setting `t1` as a float, it can be set as an array. In that case, the output will be given at the values specified at `t1` instead than at every step.

## Additional parameters

- `atol` and `rtol`: absolute and relative tolerances for the truncation error. `rtol` is set to `1e-6` and `atol` is set to 0, but it is important to provide an adequate value of `atol` for the problem. Both parameters can be arrays (with size equal to the number of equations) to set tolerances for each individual variable.

- `jacobfun`: a function evaluating the Jacobian of `odefun`. Since `geode` uses implicit methods, the solution of stiff problems require the computation of the Jacobian. If the Jacobian is not provided, it is computed by finite differences. The numerical computation of the Jacobian can become very expensive, especially if the computation of `odefun` is expensive. If the problem is not stiff, there is no need to provide the Jacobian.

- `method`: `geode` is a hybrid method, using both Adams-Moulton and BDF methods. The user may specify `method='am'` to use only Adams-Moulton method, `method='bdf'` to use only BDF, or `method='hybrid'` (which is the default) to use both methods.

- `minstep` and `maxstep`: the minimum and maximum admissible step sizes. In general, there is no reason to use these parameters. Setting a minimum step may prevent the integrator of reaching the desired accuracy, and setting a maximum step may lead to exagerated accuracy, hindering efficiency.

- `initstep`: initial step size. Setting this parameter requires previous knowledge of the behavior of the solution. If it is not provided, it will be calculated to ensure that the initial truncation error is within tolerance, usually leading to a undersized initial step size.

- `starting_method`: method to be used in the first step. By default, it is Adams-Moulton.

- `full_output`: if `full_output=True`, the truncation error, tolerance, order, method and success status of each step will be returned as arrays, which is useful for debugging.

- `verbosity`: defines what `geode` will print on the screen. If -1, it will print nothing. If 0, it will print warning and error messages. Otherwise, it will print information about the integration every `verbosity` steps.

## Implementation details
Adams-Moulton and BDF are multivalue methods, in which the evaluation of a new step uses information from previous step. In `geode`, this is made through the use of the Nordsieck matrix, allowing for easy and inexpensive change of method, order and step size. This allows also for accurate and inexpensive interpolation of the solution at desired values if `t1` is set as an array.

`geode` changes method by comparing the current step size with the step size that would be used by the other method to reach accuracy. This comparison is made at the end of each step and is very inexpensive. The change in order is made by trying adjacent orders and selecting the one that leads to the smallest step size. The attempt at changing order is made every few steps.

The implicit equation at each step is solved using functional iteration for non-stiff problems (using Adams-Moulton) and using the good Broyden's method for stiff problems (using BDF).
