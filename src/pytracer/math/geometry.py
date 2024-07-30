import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union

from .utils import are_close


def _are_xyz_close(a, b, epsilon=1e-5):
    # This works thanks to Python's duck typing. In C++ and other languages
    # you should probably rely on function templates or something like
    return (
        are_close(a.x, b.x, epsilon=epsilon)
        and are_close(a.y, b.y, epsilon=epsilon)
        and are_close(a.z, b.z, epsilon=epsilon)
    )


def _add_xyz(a, b, return_type):
    # Ditto
    return return_type(a.x + b.x, a.y + b.y, a.z + b.z)


def _sub_xyz(a, b, return_type):
    # Ditto
    return return_type(a.x - b.x, a.y - b.y, a.z - b.z)


def _mul_scalar_xyz(scalar, xyz, return_type):
    return return_type(scalar * xyz.x, scalar * xyz.y, scalar * xyz.z)


def _get_xyz_element(self, item):
    assert (item >= 0) and (item < 3), f"wrong vector index {item}"

    if item == 0:
        return self.x
    elif item == 1:
        return self.y

    return self.z


@dataclass
class Vector3D:
    """A 3D vector.

    This class has three floating-point fields: `x`, `y`, and `z`."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def is_close(self, other, epsilon=1e-5):
        """Return True if the object and 'other' have roughly the same direction and orientation"""
        assert isinstance(other, Vector3D)
        return _are_xyz_close(self, other, epsilon=epsilon)

    def __add__(self, other):
        """Sum two vectors, or one vector and one point"""
        if isinstance(other, Vector3D):
            return _add_xyz(self, other, Vector3D)
        elif isinstance(other, Point):
            return _add_xyz(self, other, Point)
        else:
            raise TypeError(
                f"Unable to run Vector3D.__add__ on a {type(self)} and a {type(other)}."
            )

    def __sub__(self, other):
        """Subtract one vector from another"""
        if isinstance(other, Vector3D):
            return _sub_xyz(self, other, Vector3D)
        else:
            raise TypeError(
                f"Unable to run Vector3D.__sub__ on a {type(self)} and a {type(other)}."
            )

    def __mul__(self, scalar):
        """Compute the product between a vector and a scalar"""
        return _mul_scalar_xyz(scalar=scalar, xyz=self, return_type=Vector3D)

    def __getitem__(self, item):
        """Return the i-th component of a vector, starting from 0"""
        return _get_xyz_element(self, item)

    def __neg__(self):
        """Return the reversed vector"""
        return Vector3D(-self.x, -self.y, -self.z)

    def __eq__(self, other):
        """Check whether two vectors are equal"""
        if isinstance(other, Vector3D):
            return _are_xyz_close(self, other)
        elif isinstance(other, (tuple, list)):
            if not (len(other) == 3):
                return False
            return _are_xyz_close(self, Vector3D(*other))
        elif isinstance(other, np.ndarray):
            if not (other.shape == (3,)):
                return False
            return _are_xyz_close(self, Vector3D(*other))
        return False

    def __ne__(self, other):
        """Check whether two vectors are different"""
        return not self.__eq__(other)

    def dot(self, other):
        """Compute the dot product between two vectors"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def squared_norm(self):
        """Return the squared norm (Euclidean length) of a vector

        This is faster than `Vector3D.norm` if you just need the squared norm."""
        return self.x**2 + self.y**2 + self.z**2

    def norm(self):
        """Return the norm (Euclidean length) of a vector"""
        return math.sqrt(self.squared_norm())

    def cross(self, other):
        """Compute the cross (outer) product between two vectors"""
        return Vector3D(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
        )

    def normalize(self):
        """Modify the vector's norm so that it becomes equal to 1"""
        norm = self.norm()
        self.x /= norm
        self.y /= norm
        self.z /= norm
        return self

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Point:
    """A point in 3D space

    This class has three floating-point fields: `x`, `y`, and `z`."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def is_close(self, other, epsilon=1e-5):
        """Return True if the object and 'other' have roughly the same position"""
        assert isinstance(other, Point)
        return _are_xyz_close(self, other, epsilon=epsilon)

    def __add__(self, other):
        """Sum a point and a vector"""
        if isinstance(other, Vector3D):
            return _add_xyz(self, other, Point)
        else:
            raise TypeError(
                f"Unable to run Point.__add__ on a {type(self)} and a {type(other)}."
            )

    def __sub__(self, other):
        """Subtract a vector from a point"""
        if isinstance(other, Vector3D):
            return _sub_xyz(self, other, Point)
        elif isinstance(other, Point):
            return _sub_xyz(self, other, Vector3D)
        else:
            raise TypeError(
                f"Unable to run __sub__ on a {type(self)} and a {type(other)}."
            )

    def __mul__(self, scalar):
        """Multiply the point by a scalar value"""
        return _mul_scalar_xyz(scalar=scalar, xyz=self, return_type=Point)

    def __getitem__(self, item):
        """Return the i-th component of a point, starting from 0"""
        return _get_xyz_element(self, item)

    def __eq__(self, other):
        """Check whether two points are equal"""
        if isinstance(other, Point):
            return _are_xyz_close(self, other)
        elif isinstance(other, (tuple, list)):
            if not (len(other) == 3):
                return False
            return _are_xyz_close(self, Point(*other))
        elif isinstance(other, np.ndarray):
            if not (other.shape == (3,)):
                return False
            return _are_xyz_close(self, Point(*other))
        return False

    def __ne__(self, other):
        """Check whether two points are different"""
        return not self.__eq__(other)

    def to_vector(self):
        """Convert a `Point` into a `Vector3D`"""
        return Vector3D(self.x, self.y, self.z)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Normal:
    """A normal vector in 3D space

    This class has three floating-point fields: `x`, `y`, and `z`."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __neg__(self):
        return Normal(-self.x, -self.y, -self.z)

    def __mul__(self, scalar):
        """Compute the product between a vector and a scalar"""
        return _mul_scalar_xyz(scalar=scalar, xyz=self, return_type=Normal)

    def is_close(self, other, epsilon=1e-5):
        """Return True if the object and 'other' have roughly the same direction and orientation"""
        assert isinstance(other, Normal)
        return _are_xyz_close(self, other, epsilon=epsilon)

    def to_vec(self) -> Vector3D:
        """Convert a normal into a :class:`Vec` type"""
        return Vector3D(self.x, self.y, self.z)

    def squared_norm(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self):
        return math.sqrt(self.squared_norm())

    def normalize(self):
        """Modify the vector's norm so that it becomes equal to 1"""
        norm = self.norm()
        self.x /= norm
        self.y /= norm
        self.z /= norm
        return self

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


VEC_X = Vector3D(1.0, 0.0, 0.0)
VEC_Y = Vector3D(0.0, 1.0, 0.0)
VEC_Z = Vector3D(0.0, 0.0, 1.0)


@dataclass
class Vector2D:
    """A 2D vector used to represent a point on a surface

    The fields are named `u` and `v` to distinguish them from the usual 3D coordinates `x`, `y`, `z`.
    """

    u: float = 0.0
    v: float = 0.0

    def is_close(self, other: "Vector2D", epsilon=1e-5):
        """Check whether two `Vector2D` points are roughly the same or not"""
        return (abs(self.u - other.u) < epsilon) and (abs(self.v - other.v) < epsilon)

    def to_array(self) -> np.ndarray:
        return np.array([self.u, self.v])


def create_onb_from_z(
    normal: Union[Vector3D, Normal]
) -> Tuple[Vector3D, Vector3D, Vector3D]:
    """Create a orthonormal basis (ONB) from a vector representing the z axis (normalized)

    Return a tuple containing the three vectors (e1, e2, e3) of the basis. The result is such
    that e3 = normal.

    The `normal` vector must be *normalized*, otherwise this method won't work.
    """
    sign = 1.0 if (normal.z > 0.0) else -1.0
    a = -1.0 / (sign + normal.z)
    b = normal.x * normal.y * a

    e1 = Vector3D(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x)
    e2 = Vector3D(b, sign + normal.y * normal.y * a, -normal.y)

    return e1, e2, Vector3D(normal.x, normal.y, normal.z)


def normalized_dot(v1: Union[Vector3D, Normal], v2: Union[Vector3D, Normal]) -> float:
    """Apply the dot product to the two arguments after having normalized them.

    The result is the cosine of the angle between the two vectors/normals."""

    # This is not terribly efficient, but we're writing in Python. You should use your
    # language's facilities (e.g., C++ templates) to make this function work seamlessly
    # with vectors *and* normals.
    v1_vec = Vector3D(v1.x, v1.y, v1.z).normalize()
    v2_vec = Vector3D(v2.x, v2.y, v2.z).normalize()

    return v1_vec.dot(v2_vec)
