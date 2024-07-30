import unittest
import math
from pytracer.math.quaternion import Quaternion


class QuaternionTests(unittest.TestCase):
    def test_creation(self):
        # Uses (w, x, y, z) notation
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q.x, 1)
        self.assertEqual(q.y, 2)
        self.assertEqual(q.z, 3)
        self.assertEqual(q.w, 4)

    def test_identity(self):
        q = Quaternion.Identity()
        self.assertEqual(q.x, 0)
        self.assertEqual(q.y, 0)
        self.assertEqual(q.z, 0)
        self.assertEqual(q.w, 1)

    def test_addition(self):
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        q3 = q1 + q2
        self.assertEqual(q3.x, 6)
        self.assertEqual(q3.y, 8)
        self.assertEqual(q3.z, 10)
        self.assertEqual(q3.w, 12)

    def test_subtraction(self):
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        q3 = q1 - q2
        self.assertEqual(q3.w, -4)
        self.assertEqual(q3.x, -4)
        self.assertEqual(q3.y, -4)
        self.assertEqual(q3.z, -4)

    def test_multiplication(self):
        # Uses (w, x, y, z) notation
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        q3 = q1 * q2
        self.assertEqual(q3.x, 24)
        self.assertEqual(q3.y, 48)
        self.assertEqual(q3.z, 48)
        self.assertEqual(q3.w, -6)

        q1 = Quaternion(1, 1, 1, 1)
        q2 = Quaternion(-1, -1, -1, 1)
        q3 = q1 * q2
        self.assertEqual(q3.x, 0)
        self.assertEqual(q3.y, 0)
        self.assertEqual(q3.z, 0)
        self.assertEqual(q3.w, 4)

        # Rotation of 90 degrees about the x-axis
        q1 = Quaternion(math.sqrt(2) / 2, 0.0, 0.0, math.sqrt(2) / 2)
        v = [0, 1, 0]
        qv = q1 * v
        self.assertAlmostEqual(qv[0], 0.0)
        self.assertAlmostEqual(qv[0], 0.0)
        self.assertAlmostEqual(qv[2], 1.0)

        # Rotation of 90 degrees about the y-axis
        q1 = Quaternion(0.0, math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2)
        v = [1, 0, 0]
        qv = q1 * v
        self.assertAlmostEqual(qv[0], 0.0)
        self.assertAlmostEqual(qv[1], 0.0)
        self.assertAlmostEqual(qv[2], -1.0)

        q1 = Quaternion(0.0, 0.0, math.sqrt(2) / 2, math.sqrt(2) / 2)
        v = [1, 0, 0]
        qv = q1 * v
        self.assertAlmostEqual(qv[0], 0.0)
        self.assertAlmostEqual(qv[1], 1.0)
        self.assertAlmostEqual(qv[2], 0.0)


if __name__ == "__main__":
    unittest.main()
