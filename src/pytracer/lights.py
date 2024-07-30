from dataclasses import dataclass
from abc import ABC, abstractmethod
from .color import Color
from .math.geometry import Point


class Light(ABC):
    """A generic light source

    This is an abstract class, and you should only use it to derive concrete classes.
    """

    def __init__(
        self,
        position: Point = Point(0, 0, 0),
        color: Color = Color(1, 1, 1),
        linear_radius: float = 0.0,
    ) -> None:
        self.position = position
        self.color = color
        self.linear_radius = linear_radius


class PointLight(Light):
    """A point light (used by the point-light renderer)

    This class holds information about a point light (a Dirac's delta in the rendering equation). The class has
    the following fields:

    -   `position`: a :class:`Point` object holding the position of the point light in 3D space
    -   `color`: the color of the point light (an instance of :class:`.Color`)
    -   `linear_radius`: a floating-point number. If non-zero, this «linear radius» `r` is used to compute the solid
        angle subtended by the light at a given distance `d` through the formula `(r / d)²`.
    """

    def __init__(
        self, position: Point, color: Color, linear_radius: float = 0.0
    ) -> None:
        super().__init__(position, color, linear_radius)
