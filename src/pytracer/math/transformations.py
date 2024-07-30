# -*- encoding: utf-8 -*-

import numpy as np
from math import sin, cos, radians
from .geometry import Vector3D, Point, Normal
from .quaternion import Quaternion


def _are_matr_close(m1, m2):
    for i in range(4):
        if not np.isclose(m1[i], m2[i]).all():
            return False
    return True


def _diff_of_products(a: float, b: float, c: float, d: float):
    # On systems supporting the FMA instruction (e.g., C++, Julia), you
    # might want to implement this function using the trick explained here:
    #
    # https://pharr.org/matt/blog/2019/11/03/difference-of-floats.html
    #
    # as it prevents roundoff errors.

    return a * b - c * d


IDENTITY_MATR4x4 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class Transformation:
    """An affine transformation.

    This class encodes an affine transformation. It has been designed with the aim of making the calculation
    of the inverse transformation particularly efficient.
    """

    def __init__(self, m=IDENTITY_MATR4x4, invm=IDENTITY_MATR4x4):
        self.m = m
        self.invm = invm

    def __mul__(self, other):
        if isinstance(other, Vector3D):
            row0, row1, row2, row3 = self.m
            return Vector3D(
                x=other.x * row0[0] + other.y * row0[1] + other.z * row0[2],
                y=other.x * row1[0] + other.y * row1[1] + other.z * row1[2],
                z=other.x * row2[0] + other.y * row2[1] + other.z * row2[2],
            )
        elif isinstance(other, Point):
            row0, row1, row2, row3 = self.m
            p = Point(
                x=other.x * row0[0] + other.y * row0[1] + other.z * row0[2] + row0[3],
                y=other.x * row1[0] + other.y * row1[1] + other.z * row1[2] + row1[3],
                z=other.x * row2[0] + other.y * row2[1] + other.z * row2[2] + row2[3],
            )
            w = other.x * row3[0] + other.y * row3[1] + other.z * row3[2] + row3[3]

            if w == 1.0:
                return p
            else:
                return Point(p.x / w, p.y / w, p.z / w)
        elif isinstance(other, Normal):
            row0, row1, row2, _ = self.invm
            return Normal(
                x=other.x * row0[0] + other.y * row1[0] + other.z * row2[0],
                y=other.x * row0[1] + other.y * row1[1] + other.z * row2[1],
                z=other.x * row0[2] + other.y * row1[2] + other.z * row2[2],
            )
        elif isinstance(other, Transformation):
            result_m = np.matmul(self.m, other.m)  # _matr_prod(self.m, other.m) n
            result_invm = np.matmul(
                other.invm, self.invm
            )  # Reverse order! (A B)^-1 = B^-1 A^-1
            return Transformation(m=result_m, invm=result_invm)
        else:
            raise TypeError(
                f"Invalid type {type(other)} multiplied to a Transformation object"
            )

    def is_consistent(self):
        """Check the internal consistency of the transformation.

        This method is useful when writing tests."""
        prod = np.matmul(self.m, self.invm)
        return _are_matr_close(prod, IDENTITY_MATR4x4)

    def __repr__(self):
        row0, row1, row2, row3 = self.m
        fmtstring = "   [{0:6.3e} {1:6.3e} {2:6.3e} {3:6.3e}],\n"
        result = "[\n"
        result += fmtstring.format(*row0)
        result += fmtstring.format(*row1)
        result += fmtstring.format(*row2)
        result += fmtstring.format(*row3)
        result += "]"
        return result

    def __str__(self):
        return self.__repr__()

    def is_close(self, other):
        """Check if `other` represents the same transform."""
        return _are_matr_close(self.m, other.m) and _are_matr_close(
            self.invm, other.invm
        )

    def inverse(self):
        """Return a `Transformation` object representing the inverse affine transformation.

        This method is very cheap to call."""
        return Transformation(m=self.invm, invm=self.m)

    def __eq__(self, other):
        if not isinstance(other, Transformation):
            return False
        return self.is_close(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    ### Static methods
    @staticmethod
    def from_translation(*args) -> "Transformation":
        """Return a :class:`.Transformation` object encoding a rigid translation

        The parameter `vec` specifies the amount of shift to be applied along the three axes.

        Args:
            vec (Union[Vector3D, Tuple[float, float, float]]): The translation vector.

        Returns:
            Transformation: The transformation object encoding the translation.
        """
        if len(args) == 1:
            vec = args[0]
        elif len(args) == 3:
            vec = Vector3D(*args)
        else:
            raise ValueError(
                "The input of Transformation.from_translation must be a Vector3D or three floats"
            )
        return Transformation(
            m=np.array(
                [
                    [1.0, 0.0, 0.0, vec.x],
                    [0.0, 1.0, 0.0, vec.y],
                    [0.0, 0.0, 1.0, vec.z],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            invm=np.array(
                [
                    [1.0, 0.0, 0.0, -vec.x],
                    [0.0, 1.0, 0.0, -vec.y],
                    [0.0, 0.0, 1.0, -vec.z],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    @staticmethod
    def from_scaling(*args) -> "Transformation":
        """Return a :class:`.Transformation` object encoding a scaling

        The parameter `vec` specifies the amount of scaling along the three directions X, Y, Z.

        Args:
            vec (Union[Vector3D, Tuple[float, float, float]]): The scaling factors along the three axes.

        Returns:
            Transformation: The transformation object encoding the scaling.
        """
        if len(args) == 1:
            vec = args[0]
        elif len(args) == 3:
            vec = Vector3D(*args)
        else:
            raise ValueError(
                "The input of Transformation.from_scaling must be a Vector3D or three floats"
            )
        return Transformation(
            m=np.array(
                [
                    [vec.x, 0.0, 0.0, 0.0],
                    [0.0, vec.y, 0.0, 0.0],
                    [0.0, 0.0, vec.z, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            invm=np.array(
                [
                    [1 / vec.x, 0.0, 0.0, 0.0],
                    [0.0, 1 / vec.y, 0.0, 0.0],
                    [0.0, 0.0, 1 / vec.z, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    @staticmethod
    def from_rotation_x(angle_deg: float) -> "Transformation":
        """Return a :class:`.Transformation` object encoding a rotation around the X axis

        The parameter `angle_deg` specifies the rotation angle (in degrees). The positive sign is
        given by the right-hand rule.

        Args:
            angle_deg (float): The rotation angle in degrees.

        Returns:
            Transformation: The transformation object encoding the rotation.
        """

        sinang, cosang = sin(radians(angle_deg)), cos(radians(angle_deg))
        return Transformation(
            m=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, cosang, -sinang, 0.0],
                    [0.0, sinang, cosang, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            invm=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, cosang, sinang, 0.0],
                    [0.0, -sinang, cosang, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    @staticmethod
    def from_rotation_y(angle_deg: float):
        """Return a :class:`.Transformation` object encoding a rotation around the Y axis

        The parameter `angle_deg` specifies the rotation angle (in degrees). The positive sign is
        given by the right-hand rule."""
        sinang, cosang = sin(radians(angle_deg)), cos(radians(angle_deg))
        return Transformation(
            m=np.array(
                [
                    [cosang, 0.0, sinang, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-sinang, 0.0, cosang, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            invm=np.array(
                [
                    [cosang, 0.0, -sinang, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [sinang, 0.0, cosang, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    @staticmethod
    def from_rotation_z(angle_deg: float):
        """Return a :class:`.Transformation` object encoding a rotation around the Z axis

        The parameter `angle_deg` specifies the rotation angle (in degrees). The positive sign is
        given by the right-hand rule."""
        sinang, cosang = sin(radians(angle_deg)), cos(radians(angle_deg))
        return Transformation(
            m=np.array(
                [
                    [cosang, -sinang, 0.0, 0.0],
                    [sinang, cosang, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            invm=np.array(
                [
                    [cosang, sinang, 0.0, 0.0],
                    [-sinang, cosang, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    @staticmethod
    def look_at(position: Point, target: Point, up: Vector3D) -> "Transformation":
        """Return a :class:`.Transformation` object encoding a camera transformation

        The camera is placed at the `position` point, looking at the `target` point. The `up` vector
        specifies the direction of the camera's up axis.
        """
        forward = (target - position).normalize()
        right = forward.cross(up).normalize()
        new_up = right.cross(forward)

        m = np.array(
            [
                [right.x, right.y, right.z, 0.0],
                [new_up.x, new_up.y, new_up.z, 0.0],
                [-forward.x, -forward.y, -forward.z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        invm = np.array(
            [
                [right.x, new_up.x, -forward.x, 0.0],
                [right.y, new_up.y, -forward.y, 0.0],
                [right.z, new_up.z, -forward.z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        return Transformation(m=m, invm=invm)

    @staticmethod
    def extract_translation(m: np.ndarray) -> Vector3D:
        """Extract the translation part of a transformation"""
        if not isinstance(m, np.ndarray):
            raise TypeError(
                "The input of Transformation.extract_translation must be a numpy array"
            )
        if m.shape != (4, 4):
            raise ValueError(
                "The input of Transformation.extract_translation must be a 4x4 matrix"
            )
        return Vector3D(m[0, 3], m[1, 3], m[2, 3])

    @staticmethod
    def extract_scaling(m: np.ndarray) -> Vector3D:
        """Extract the scaling part of a transformation"""
        if not isinstance(m, np.ndarray):
            raise TypeError(
                "The input of Transformation.extract_scaling must be a numpy array"
            )
        if m.shape != (4, 4):
            raise ValueError(
                "The input of Transformation.extract_scaling must be a 4x4 matrix"
            )
        sx = np.linalg.norm(m[0, :3])
        sy = np.linalg.norm(m[1, :3])
        sz = np.linalg.norm(m[2, :3])
        return Vector3D(sx, sy, sz)

    @staticmethod
    def extract_rotation(m: np.ndarray) -> Quaternion:
        """Extract the rotation part of a transformation"""
        if not isinstance(m, np.ndarray):
            raise TypeError(
                "The input of Transformation.extract_rotation must be a numpy array"
            )
        if m.shape != (4, 4):
            raise ValueError(
                "The input of Transformation.extract_rotation must be a 4x4 matrix"
            )
        sx = np.linalg.norm(m[0, :3])
        sy = np.linalg.norm(m[1, :3])
        sz = np.linalg.norm(m[2, :3])
        mq = np.array(
            [
                [m[0, 0] / sx, m[0, 1] / sy, m[0, 2] / sz, 0.0],
                [m[1, 0] / sx, m[1, 1] / sy, m[1, 2] / sz, 0.0],
                [m[2, 0] / sx, m[2, 1] / sy, m[2, 2] / sz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return Quaternion.from_matrix(mq)

    @staticmethod
    def decompose(m: np.ndarray) -> tuple:
        """Decompose a transformation into its translation, rotation, and scaling components"""
        if not isinstance(m, np.ndarray):
            raise TypeError(
                "The input of Transformation.decompose must be a numpy array"
            )
        if m.shape != (4, 4):
            raise ValueError(
                "The input of Transformation.decompose must be a 4x4 matrix"
            )
        translation = Transformation.extract_translation(m)
        m[0, 3] = m[1, 3] = m[2, 3] = 0.0
        scaling = Transformation.extract_scaling(m)
        rotation = Transformation.extract_rotation(m)
        return translation, rotation, scaling
