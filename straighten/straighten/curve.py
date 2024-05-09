import warnings
from typing import Union, Sequence, Callable

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d

ShapeLike = Union[int, Sequence[int]]


def frenet_serret(*gradients):
    _, d = gradients[0].shape
    basis = []
    for grad in gradients:
        e = grad
        # Gram-Schmidt process
        for v in basis:
            e = e - v * (v * grad).sum(axis=-1, keepdims=True)

        e /= np.linalg.norm(e, axis=-1, keepdims=True)
        basis.append(e)

    return np.stack(basis, -1)


class Interpolator:
    def __init__(self, curve: np.ndarray, step: float,
                 spacing: Union[float, Sequence[Union[float, Sequence[float]]]] = 1,
                 get_local_basis: Callable = frenet_serret):
        """
        Parameters
        ----------
        curve: array of shape (n_points, dim)
        step: step size along the curve
        spacing: the nd-pixel spacing
        get_local_basis: Callable(*gradients) -> local_basis
        """
        if curve.ndim != 2:
            raise ValueError(f'The curve shape must be (n_points, dim), but {curve.shape} provided.')
        dim = curve.shape[1]
        if isinstance(spacing, (int, float)):
            spacing = [spacing] * dim
        if dim != len(spacing):
            raise ValueError(f'"spacing" must contain {dim} elements, but {len(spacing)} provided.')
        if not np.isfinite(curve).all():
            raise ValueError(f'The curve must contain only finite values.')

        even_curve, *grads = get_derivatives(pixel_to_spatial(curve, spacing), step)
        self.dim = dim
        self.spacing = spacing
        self.knots = even_curve
        self.basis = get_local_basis(*grads)

    def get_grid(self, shape: ShapeLike):
        """
        Make the grid onto which images will be interpolated.

        Parameters
        ----------
        shape:
            the desired shape of the image

        Returns
        -------
        grid: (dim, n_points, *shape)
        """
        shape = np.broadcast_to(shape, self.dim - 1)
        grid = np.meshgrid(*(np.arange(s) - s / 2 for s in shape))
        zs = np.zeros_like(grid[0])
        grid = np.stack([zs, *grid])

        grid = np.einsum('Nij,j...->Ni...', self.basis, grid)
        grid = np.moveaxis(grid, [0, 1], [-2, -1])
        grid = spatial_to_pixel(grid + self.knots, self.spacing)
        return np.moveaxis(grid, [-2, -1], [1, 0])

    def interpolate_along(self, array, shape: ShapeLike, fill_value: Union[float, Callable] = 0, order: int = 1):
        """
        Interpolates an incoming image along the curve.

        Parameters
        ----------
        array:
            the image to be interpolated
        shape:
            the output shape of the image along all the axes, except the first one
        fill_value:
            the value to be used outside of actual image limits
        order:
            the order of interpolation

        Examples
        --------
        >>> inter = Interpolator(curve, 1, 1)
        # `image` is a 3d image
        >>> x = inter.interpolate_along(image, (50, 50))
        >>> x.shape
        (28, 50, 50) # 28 is the number of points taken with step 1 along the curve
        """
        if callable(fill_value):
            fill_value = fill_value(array)
        return map_coordinates(array, self.get_grid(shape), order=order, cval=fill_value)

    def global_to_local(self, points, shape: ShapeLike):
        """
        Converts the coordinates from image space to local coordinates along the curve.
        """
        return self._transform(pixel_to_spatial(self._check_points(points), self.spacing), shape, self._to_local)

    def local_to_global(self, points, shape: ShapeLike):
        """
        Converts the coordinates from local coordinates along the curve to image space.
        """
        return spatial_to_pixel(self._transform(self._check_points(points), shape, self._to_global), self.spacing)

    def _get_centers(self, shape):
        centers = np.zeros_like(self.knots)
        centers[:, 0] = cumulative_length(self.knots)
        centers[:, 1:] = shape / 2
        return centers

    def _to_local(self, point, shape):
        points = point - self.knots
        to_origin = np.linalg.norm(points, axis=-1)

        points = np.einsum('nji,nj->ni', self.basis, points)
        to_plane = points[:, 0]

        return interpolate_coords(points + self._get_centers(shape), to_origin, to_plane)

    def _to_global(self, point, shape):
        points = point - self._get_centers(shape)
        to_plane = points[:, 0]

        points = np.einsum('nij,nj->ni', self.basis, points)
        to_origin = np.linalg.norm(points, axis=-1)

        return interpolate_coords(points + self.knots, to_origin, to_plane)

    @staticmethod
    def _transform(points, shape, func):
        # point: *any, dim
        *spatial, d = points.shape
        shape = np.broadcast_to(shape, d - 1)
        points = points.reshape(-1, d)
        results = []
        for p in points:
            results.append(func(p, shape))

        return np.array(results).reshape(*spatial, d)

    def _check_points(self, points):
        points = np.asarray(points)
        d = points.shape[-1]
        if d != self.dim:
            raise ValueError(f"The points dim ({d}) doesn't match the curve dim ({self.dim}).")
        return points


def pixel_to_spatial(points, spacing, v=False):
    with np.errstate(divide='raise', invalid='raise'):
        points = np.asarray(points)
        if not points.size:
            return points

        _check_dim_consistency(points.shape, spacing)
        result = []
        for i, sp in enumerate(spacing):
            axis = points[..., i]
            if isinstance(sp, (int, float)):
                axis = axis * sp
            else:
                axis = interp1d(np.arange(len(sp)), sp, bounds_error=False, fill_value='extrapolate')(axis)

            result.append(axis)
        return np.stack(result, -1)


def spatial_to_pixel(points, spacing):
    with np.errstate(divide='raise', invalid='raise'):
        points = np.asarray(points)
        if not points.size:
            return points

        _check_dim_consistency(points.shape, spacing)
        result = []
        for i, sp in enumerate(spacing):
            axis = points[..., i]
            if isinstance(sp, (int, float)):
                axis = axis / sp
            else:
                axis = interp1d(sp, np.arange(len(sp)), bounds_error=False, fill_value='extrapolate')(axis)

            result.append(axis)
        return np.stack(result, -1)


def _check_dim_consistency(shape, spacing):
    if shape[-1] != len(spacing):
        raise ValueError(f"The points dim ({shape[-1]}) doesn't match the spacing size ({len(spacing)})")


def cumulative_length(curve: np.ndarray):
    lengths = np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
    lengths = np.insert(lengths, 0, 0)
    return lengths


def get_derivatives(curve: np.ndarray, step: float):
    assert curve.ndim == 2
    _, d = curve.shape

    lengths = cumulative_length(curve)
    xs = np.arange(0, lengths[-1], step)
    yield interp1d(lengths, curve, axis=0)(xs)

    grad = curve
    for _ in range(d):
        grad = np.gradient(grad, axis=0)
        yield interp1d(lengths, grad, axis=0)(xs)


def interpolate_coords(coordinates, distance_to_origin, distance_to_plane):
    idx = distance_to_origin.argmin()

    # how many good planes are there?
    candidates, = np.diff(np.sign(distance_to_plane)).nonzero()
    # ensure that there is exactly one zero
    if len(candidates) != 1:
        warnings.warn("Couldn't uniquely choose a local basis.")

    # adjust the index by the point of sign change
    if len(candidates) > 0:
        idx = candidates[np.abs(candidates - idx).argmin()]
    slc = slice(max(0, idx - 2), idx + 2)

    distance_to_plane = distance_to_plane[slc]
    coordinates = coordinates[slc]
    return interp1d(distance_to_plane, coordinates, axis=0, bounds_error=False, fill_value='extrapolate')(0)
