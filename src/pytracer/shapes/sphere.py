from typing import Union
from math import sqrt, atan2, acos, pi
from .aabb import AABB
from pytracer.math.geometry import Point, Vector3D, Vector2D, Normal
from pytracer.math.transformations import Transformation
from pytracer.ray import Ray
from pytracer.materials.material import Material
from pytracer.hitrecord import HitRecord
from pytracer.shapes.shape import Shape


def _sphere_point_to_uv(point: Point) -> Vector2D:
    """Convert a 3D point on the surface of the unit sphere into a (u, v) 2D point"""
    u = atan2(point.y, point.x) / (2.0 * pi)
    return Vector2D(
        u=u if u >= 0.0 else u + 1.0,
        v=acos(point.z) / pi,
    )


def _sphere_normal(point: Point, ray_dir: Vector3D) -> Normal:
    """Compute the normal of a unit sphere

    The normal is computed for `point` (a point on the surface of the
    sphere), and it is chosen so that it is always in the opposite
    direction with respect to `ray_dir`.

    """
    result = Normal(point.x, point.y, point.z)
    return result if (point.to_vector().dot(ray_dir) < 0.0) else -result


class Sphere(Shape):
    """A 3D unit sphere centered on the origin of the axes"""

    def __init__(
        self, transformation=Transformation(), material: Material = Material()
    ):
        """Create a unit sphere, potentially associating a transformation to it"""
        super().__init__(transformation, material)

    def compute_aabb(self) -> None:
        """Compute the axis-aligned bounding box of the sphere
        The AABB of a sphere is a cube with the sphere's diameter as the side length, centered on the sphere's center.
        The minimum point of the AABB is the sphere's center minus the radius, and the maximum point is the sphere's center plus the radius.
        """
        self.aabb = AABB()

    def ray_intersection(self, ray: Ray) -> Union[HitRecord, None]:
        """Checks if a ray intersects the sphere

        Return a `HitRecord`, or `None` if no intersection was found.
        """
        inv_ray = ray.transform(self.transformation.inverse())
        origin_vec = inv_ray.origin.to_vector()
        a = inv_ray.dir.squared_norm()
        b = 2.0 * origin_vec.dot(inv_ray.dir)
        c = origin_vec.squared_norm() - 1.0

        delta = b * b - 4.0 * a * c
        if delta <= 0.0:
            return None

        sqrt_delta = sqrt(delta)
        tmin = (-b - sqrt_delta) / (2.0 * a)
        tmax = (-b + sqrt_delta) / (2.0 * a)

        if (tmin > inv_ray.tmin) and (tmin < inv_ray.tmax):
            first_hit_t = tmin
        elif (tmax > inv_ray.tmin) and (tmax < inv_ray.tmax):
            first_hit_t = tmax
        else:
            return None

        hit_point = inv_ray.at(first_hit_t)
        return HitRecord(
            world_point=self.transformation * hit_point,
            normal=self.transformation * _sphere_normal(hit_point, inv_ray.dir),
            surface_point=_sphere_point_to_uv(hit_point),
            t=first_hit_t,
            ray=ray,
            material=self.material,
        )

    def quick_ray_intersection(self, ray: Ray) -> bool:
        """Quickly checks if a ray intersects the sphere"""
        inv_ray = ray.transform(self.transformation.inverse())
        origin_vec = inv_ray.origin.to_vector()
        a = inv_ray.dir.squared_norm()
        b = 2.0 * origin_vec.dot(inv_ray.dir)
        c = origin_vec.squared_norm() - 1.0

        delta = b * b - 4.0 * a * c
        if delta <= 0.0:
            return False

        sqrt_delta = sqrt(delta)
        tmin = (-b - sqrt_delta) / (2.0 * a)
        tmax = (-b + sqrt_delta) / (2.0 * a)

        return (inv_ray.tmin < tmin < inv_ray.tmax) or (
            inv_ray.tmin < tmax < inv_ray.tmax
        )
