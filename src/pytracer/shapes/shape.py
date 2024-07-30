from typing import Union
from abc import ABC, abstractmethod
from pytracer.hitrecord import HitRecord
from pytracer.ray import Ray
from pytracer.math.transformations import Transformation
from pytracer.materials.material import Material
from .aabb import AABB


class Shape(ABC):
    """A generic 3D shape

    This is an abstract class, and you should only use it to derive
    concrete classes. Be sure to redefine the method
    :meth:`.Shape.ray_intersection`.

    """

    def __init__(
        self,
        transformation: Transformation = Transformation(),
        material: Material = Material(),
    ):
        """Create a shape, potentially associating a transformation to it"""
        self.transformation = transformation
        self.material = material
        self.compute_aabb()

    @abstractmethod
    def compute_aabb(self) -> None:
        """Compute the axis-aligned bounding box of the shape"""
        raise NotImplementedError(
            "Shape.compute_aabb is an abstract method and cannot be called directly"
        )

    @abstractmethod
    def ray_intersection(self, ray: Ray) -> Union[HitRecord, None]:
        """Compute the intersection between a ray and this shape"""
        raise NotImplementedError(
            "Shape.ray_intersection is an abstract method and cannot be called directly"
        )

    @abstractmethod
    def quick_ray_intersection(self, ray: Ray) -> bool:
        """Determine whether a ray hits the shape or not"""
        raise NotImplementedError(
            "Shape.quick_ray_intersection is an abstract method and cannot be called directly"
        )
