from math import floor
from typing import Union
from pytracer.materials.material import Material
from pytracer.math.geometry import Normal, Vector2D, Point
from pytracer.math.transformations import Transformation
from pytracer.shapes.shape import Shape
from pytracer.shapes.aabb import AABB
from pytracer.hitrecord import HitRecord
from pytracer.ray import Ray


class Plane(Shape):
    """A 3D infinite plane parallel to the x and y axis and passing through the origin."""

    def __init__(
        self, transformation=Transformation(), material: Material = Material()
    ):
        """Create a xy plane, potentially associating a transformation to it"""
        super().__init__(transformation, material)

    def compute_aabb(self) -> None:
        """Compute the axis-aligned bounding box of the plane.
        The AABB of a plane is a cube with side length 2, centered on the origin.
        The minimum point of the AABB is (-1, -1, 0), and the maximum point is (1, 1, 0).
        """

        self.aabb = AABB(Point(-1, -1, 0), Point(1, 1, 0))

    def ray_intersection(self, ray: Ray) -> Union[HitRecord, None]:
        """Checks if a ray intersects the plane

        Return a `HitRecord`, or `None` if no intersection was found.
        """
        inv_ray = ray.transform(self.transformation.inverse())
        if abs(inv_ray.dir.z) < 1e-5:
            return None

        t = -inv_ray.origin.z / inv_ray.dir.z

        if (t <= inv_ray.tmin) or (t >= inv_ray.tmax):
            return None

        hit_point = inv_ray.at(t)

        return HitRecord(
            world_point=self.transformation * hit_point,
            normal=self.transformation
            * Normal(0.0, 0.0, 1.0 if inv_ray.dir.z < 0.0 else -1.0),
            surface_point=Vector2D(
                hit_point.x - floor(hit_point.x), hit_point.y - floor(hit_point.y)
            ),
            t=t,
            ray=ray,
            material=self.material,
        )

    def quick_ray_intersection(self, ray: Ray) -> bool:
        """Quickly checks if a ray intersects the plane"""
        inv_ray = ray.transform(self.transformation.inverse())
        if abs(inv_ray.dir.z) < 1e-5:
            return False

        t = -inv_ray.origin.z / inv_ray.dir.z
        return inv_ray.tmin < t < inv_ray.tmax
