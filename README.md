# squish


A simple n-d slice sampler, using a covariance matrix (or ellipsoid) to define slice directions and step sizes

### Example

The Rosenbrock function has the density:
```python
p(x_1,x_2) = -(a-x_1)**2 - b * (x_2 - x_1**2)**2
```
which has a maximum at `x_1 = x_2 = a`.

If we sample this for 10000 iterations with `squish`,
starting at `x = (-1, 1)` with an ellipsoidal direction and step size given by `diag([a,b])` we obtain

![Rosenbrock](squish/rosenbrock.png?raw=True "Rosenbrock with squish")
