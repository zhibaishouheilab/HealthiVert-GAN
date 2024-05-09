Interpolation along multidimensional curves.

## Installation

```bash
git clone https://github.com/neuro-ml/straighten.git
cd straighten
pip install -e .
```

## Usage

Prepare the interpolator

```python
from straighten import Interpolator

curve = ...  # a 3d curve
# `step` is the distance along used to take equidistant points
inter = Interpolator(curve, step=1)
```

Interpolate the image

```python
image = ...  # a 3d image containing the curve
straight = inter.interpolate_along(image, shape=(80, 80))
# N - is the number of equidistant points along the curve
print(straight.shape)  # (N, 80, 80)
```

Move additional points from the image to the new coordinates system:

```python
# the shape must be the same as for the straightened image
local = inter.global_to_local(points, shape=(80, 80))
```

Move points from the new coordinate system back:

```python
original = inter.local_to_global(local, shape=(80, 80))
```

## Making a local basis

In order to change the coordinate system we need a local basis along the curve.
[Frenet–Serret](https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas) is a good starting point, however, for
certain curves it can give unwanted rotations along the tangential vectors.

If you need a better local basis for your task, you can pass the `get_local_basis` argument in `Interpolator`'s
constructor. E.g. see the implementation of [frenet_serret](straighten/curve.py).

For example, in this [paper](https://arxiv.org/abs/2005.11960) we used the following local basis:

```python
import numpy as np


def get_local_basis(grad, *args):
    grad = grad / np.linalg.norm(grad, axis=1, keepdims=True)

    # the second basis vector must be in the sagittal plane
    sagittal = grad[:, [0, 2]]
    second = sagittal[:, ::-1] * [1, -1]

    # choose the right orientation of the basis (avoiding rotations)
    dets = np.linalg.det(np.stack([sagittal, second], -1))
    second = second * dets[:, None]
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    second = np.insert(second, 1, np.zeros_like(second[:, 0]), axis=1)

    third = np.cross(second, grad)

    return np.stack([grad, second, third], -1)
```

which yields a much more stable local basis for our particular task.
