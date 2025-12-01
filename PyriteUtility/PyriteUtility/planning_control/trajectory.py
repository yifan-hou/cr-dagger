"""
This file is modified based on Huang Pham's Toppra package
    https://github.com/hungpham2511/toppra/

Copying license below
------
MIT License

Copyright (c) 2017 Hung Pham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from typing import List, Union
import typing as T
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.interpolate import interp1d

from spatialmath import SE3, SO3, UnitQuaternion, Quaternion
import spatialmath.base as smb


class AbstractGeometricPath(object):
    """Abstract base class that represents geometric paths.

    Derive geometric paths classes should implement the below abstract
    methods. These methods are expected in different steps of the
    algorithm.

    """

    def __call__(
        self, path_positions: Union[float, np.ndarray], order: int = 0
    ) -> np.ndarray:
        """Evaluate the path at given positions.

        Parameters
        ----------
        path_positions: float or np.ndarray
            Path positions to evaluate the interpolator.
        order: int
            Order of the evaluation call.

            - 0: position
            - 1: first-order derivative
            - 2: second-order derivative

        Returns
        -------
        :
            The evaluated joint positions, velocity or
            accelerations. The shape of the result depends on the
            shape of the input, it is either (N, m) where N is the
            number of path positions and m is the number of
            degree-of-freedom, or (m,).

        """
        raise NotImplementedError

    @property
    def dof(self) -> int:
        """Return the degrees-of-freedom of the path."""
        raise NotImplementedError

    @property
    def path_interval(self):
        """Return the starting and ending path positions.

        Returns
        -------
        np.ndarray(2,)
            The starting and ending path positions.

        """
        raise NotImplementedError

    @property
    def waypoints(self):
        """Tuple[ndarray, ndarray] or None: The path's waypoints if applicable. None otherwise."""
        raise NotImplementedError

    def eval(self, ss_sam: Union[float, np.ndarray]):
        """Evaluate the path values."""
        return self(ss_sam, 0)

    def evald(self, ss_sam: Union[float, np.ndarray]):
        """Evaluate the path first-derivatives."""
        return self(ss_sam, 1)

    def evaldd(self, ss_sam: Union[float, np.ndarray]):
        """Evaluate the path second-derivatives."""
        return self(ss_sam, 2)


class CombinedGeometricPath:
    """Class that represents a combination of multiple geometric paths."""

    def __init__(
        self,
        list_of_paths: List[AbstractGeometricPath],
    ) -> None:
        self.list_of_paths = list_of_paths

    def __call__(self, path_positions, order=0):
        return [
            self.list_of_paths[i](path_positions, order)
            for i in range(len(self.list_of_paths))
        ]


class LinearInterpolator(AbstractGeometricPath):
    """Linearly interpolate the given waypoints.

    Parameters
    ----------
    x_wp: np.ndarray(m,)
        Path positions of the waypoints.
    y_wp: np.ndarray(m, d)
        Waypoints.
    """

    def __init__(
        self,
        x_wp,
        y_wp,
    ) -> None:
        super(LinearInterpolator, self).__init__()
        self.x_wp = np.array(x_wp)  # type: np.ndarray
        self.y_wp = np.array(y_wp)  # type: np.ndarray
        assert self.x_wp.shape[0] == self.y_wp.shape[0]
        self.f = interp1d(
            x=self.x_wp,
            y=self.y_wp,
            axis=0,
        )

    def __call__(self, path_positions, order=0):
        if order == 0:
            return self.f(path_positions)
        if order >= 1:
            raise NotImplementedError
        raise ValueError(f"Invalid order {order}")

    @property
    def waypoints(self):
        """Tuple[np.ndarray, np.ndarray]: Return the waypoints.

        The first element is the positions, the second element is the
        array of waypoints.

        """
        return self.x_wp, self.y_wp

    @property
    def duration(self):
        """Return the duration of the path."""
        return self.x_wp[-1] - self.x_wp[0]

    @property
    def path_interval(self):
        """Return the start and end points."""
        return np.array([self.x_wp[0], self.x_wp[-1]])

    @property
    def dof(self):
        if np.isscalar(self.y_wp[0]):
            return 1
        return self.y_wp[0].shape[0]


class SplineInterpolator(AbstractGeometricPath):
    """Interpolate the given waypoints by cubic spline.

    This interpolator is implemented as a simple wrapper over scipy's
    CubicSpline class.

    Parameters
    ----------
    ss_waypoints: np.ndarray(m,)
        Path positions of the waypoints.
    waypoints: np.ndarray(m, d)
        Waypoints.
    bc_type: optional
        Boundary conditions of the spline. Can be 'not-a-knot',
        'clamped', 'natural' or 'periodic'.

        - 'not-a-knot': The most default option, return the most naturally
          looking spline.
        - 'clamped': First-order derivatives of the spline at the two
          end are clamped at zero.

        See scipy.CubicSpline documentation for more details.

    """

    def __init__(self, ss_waypoints, waypoints, bc_type: str = "not-a-knot") -> None:
        super(SplineInterpolator, self).__init__()
        self.ss_waypoints = np.array(ss_waypoints)  # type: np.ndarray
        self._q_waypoints = np.array(waypoints)  # type: np.ndarray
        assert self.ss_waypoints.shape[0] == self._q_waypoints.shape[0]

        self.cspl: T.Union[T.Callable[[T.Any], T.Any], CubicSpline]
        self.cspld: T.Union[T.Callable[[T.Any], T.Any], CubicSpline]
        assert len(ss_waypoints) > 1
        self.cspl = CubicSpline(ss_waypoints, waypoints, bc_type=bc_type)
        self.cspld = self.cspl.derivative()
        self.cspldd = self.cspld.derivative()

    def __call__(self, path_positions, order=0):
        if order == 0:
            return self.cspl(path_positions)
        if order == 1:
            return self.cspld(path_positions)
        if order == 2:
            return self.cspldd(path_positions)
        raise ValueError(f"Invalid order {order}")

    @property
    def waypoints(self):
        """Tuple[np.ndarray, np.ndarray]: Return the waypoints.

        The first element is the positions, the second element is the
        array of waypoints.

        """
        return self.ss_waypoints, self._q_waypoints

    @property
    def duration(self):
        """Return the duration of the path."""
        return self.ss_waypoints[-1] - self.ss_waypoints[0]

    @property
    def path_interval(self):
        """Return the start and end points."""
        return np.array([self.ss_waypoints[0], self.ss_waypoints[-1]])

    @property
    def dof(self):
        if np.isscalar(self._q_waypoints[0]):
            return 1
        return self._q_waypoints[0].shape[0]


class LinearTransformationInterpolator(AbstractGeometricPath):
    """Linearly interpolate the given orientation/pose waypoints. Use SLERP for the rotation part.

    Parameters
    ----------
    x_wp: np.ndarray(m,)
        Path positions of the waypoints.
    y_wp: np.ndarray(m, d, d)
        Waypoints. d is 3 for SO3, 4 for SE3.
    """

    def __init__(
        self,
        x_wp,
        y_wp,
    ) -> None:
        super(LinearTransformationInterpolator, self).__init__()
        print("[LinearTransformationInterpolator] Warning: This class is buggy because the spatial-math package sometimes throw errors during slerp.")
        self.x_wp = np.array(x_wp)  # type: np.ndarray
        self.y_wp = np.array(y_wp)  # type: np.ndarray
        assert self.x_wp.shape[0] == self.y_wp.shape[0]
        assert self.y_wp.shape[1] == self.y_wp.shape[2]
        if self.y_wp.shape[1] == 3:
            self.transform_wp = [SO3(smb.trnorm(y), check=True) for y in y_wp]
        elif self.y_wp.shape[1] == 4:
            self.transform_wp = [SE3(smb.trnorm(y), check=True) for y in y_wp]

    def __call__(self, path_positions, order=0):
        if order >= 1:
            print(
                "[LinearTransformationInterpolator] Warning: derivatives are not implemented for rotation interpolation."
            )
            raise NotImplementedError
        if (
            np.max(path_positions) > self.x_wp[-1]
            or np.min(path_positions) < self.x_wp[0]
        ):
            raise ValueError("Path positions are out of bounds.")

        if np.isscalar(path_positions):
            path_positions = [path_positions]
        id1 = np.searchsorted(self.x_wp, path_positions)

        for i in range(len(id1)):
            if id1[i] == 0:
                id1[i] = 1
        id0 = id1 - 1
        x = (path_positions - self.x_wp[id0]) / (self.x_wp[id1] - self.x_wp[id0])

        result = []
        for i in range(len(id0)):
            temp = self.transform_wp[id0[i]].interp(self.transform_wp[id1[i]], x[i])
            if temp.data[0] is None:
                raise ValueError(
                    f"interp returns None at i={i} of {id0[i]} and {id1[i]} with x={x[i]}"
                )
            result.append(temp.data[0])
        return np.array(result)

        # # debug
        # A = self.transform_wp[0]
        # B = self.transform_wp[1]
        # C = A.interp(B, x[0])
        # print("A: ", A)
        # print("B: ", B)
        # print("C: ", C)

        return np.array(
            [
                self.transform_wp[id0[i]].interp(self.transform_wp[id1[i]], x[i])
                for i in range(len(id0))
            ]
        )

    @property
    def waypoints(self):
        """Tuple[np.ndarray, np.ndarray]: Return the waypoints.

        The first element is the positions, the second element is the
        array of waypoints.

        """
        return self.x_wp, self.y_wp

    @property
    def duration(self):
        """Return the duration of the path."""
        return self.x_wp[-1] - self.x_wp[0]

    @property
    def path_interval(self):
        """Return the start and end points."""
        return np.array([self.x_wp[0], self.x_wp[-1]])

    @property
    def dof(self):
        if np.isscalar(self.y_wp[0]):
            return 1
        return self.y_wp[0].shape[0]
