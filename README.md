# squish


A simple n-d slice sampler, using a covariance matrix (or ellipsoid) to set step sizes and draw slice directions.

### Example

The Rosenbrock function has the density:
```python
lnp(x_1,x_2) = -(a-x_1)**2 - b * (x_2 - x_1**2)**2
```
which has a maximum at `x_1 = x_2 = a`.

If we sample this for 1e6 iterations with `squish`,
starting at `x = (-1, 1)` we obtain

![Rosenbrock](squish/rosenbrock.png?raw=True "Rosenbrock with squish")

The autocorrelation times are ~300-400 in `x_1` and `x_2`, only weakly dependent on the covariance of the ellipsoid (though the number of likelihood calls changes).
