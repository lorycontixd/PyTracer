import dataclasses
from dataclasses import dataclass
from typing import Union

from .math.geometry import Point, Normal, Vector2D
from .materials.material import Material
from .ray import Ray


@dataclass
class HitRecord:
    """
    A class holding information about a ray-shape intersection

    The parameters defined in this dataclass are the following:

    -   `world_point`: a :class:`.Point` object holding the world coordinates of the hit point
    -   `normal`: a :class:`.Normal` object holding the orientation of the normal to the surface where the hit happened
    -   `surface_point`: a :class:`.Vec2d` object holding the position of the hit point on the surface of the object
    -   `t`: a floating-point value specifying the distance from the origin of the ray where the hit happened
    -   `ray`: the ray that hit the surface
    """

    world_point: Point = dataclasses.field(default_factory=Point)
    normal: Normal = dataclasses.field(default_factory=Normal)
    surface_point: Vector2D = dataclasses.field(default_factory=Vector2D)
    t: float = -1.0
    ray: Ray = dataclasses.field(default_factory=Ray)
    material: Material = dataclasses.field(default_factory=Material)

    def is_close(self, other: Union["HitRecord", None], epsilon=1e-5) -> bool:
        """Check whether two `HitRecord` represent the same hit event or not"""
        if not other:
            return False

        return (
            self.world_point.is_close(other.world_point)
            and self.normal.is_close(other.normal)
            and self.surface_point.is_close(other.surface_point)
            and (abs(self.t - other.t) < epsilon)
            and self.ray.is_close(other.ray)
        )
