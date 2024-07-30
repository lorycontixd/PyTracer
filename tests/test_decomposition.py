import unittest
from pytracer.math.transformations import Transformation
from pytracer.math.geometry import Vector3D
from pytracer.math.quaternion import Quaternion


class DecompositionTests(unittest.TestCase):
    def test_extract_traslation(self):
        v = Vector3D(1, 2, 3)
        t = Transformation.from_translation(v)

        self.assertEqual(Transformation.extract_translation(t.m), v)

    def test_extract_rotation(self):
        x = 90.0
        t = Transformation.from_rotation_x(x)

        rot_mat = Transformation.extract_rotation(t.m)
        rot_mat_from_q = Quaternion.from_angle_axis(x, Vector3D(1, 0, 0))
        print(f"rot_mat: {rot_mat},\t rot_mat_from_q: {rot_mat_from_q}")
