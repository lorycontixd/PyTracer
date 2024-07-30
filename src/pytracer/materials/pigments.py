from math import floor
from pytracer.color import Color
from pytracer.math.geometry import Vector2D
from pytracer.hdrimages import HdrImage


class Pigment:
    """A «pigment»

    This abstract class represents a pigment, i.e., a function that associates a color with
    each point on a parametric surface (u,v). Call the method :meth:`.Pigment.get_color` to
    retrieve the color of the surface given a :class:`.Vec2d` object."""

    def get_color(self, uv: Vector2D) -> Color:
        """Return the color of the pigment at the specified coordinates"""
        raise NotImplementedError(
            "Method Pigment.get_color is abstract and cannot be called"
        )


class UniformPigment(Pigment):
    """A uniform pigment

    This is the most boring pigment: a uniform hue over the whole surface."""

    def __init__(self, color=Color()):
        self.color = color

    def get_color(self, uv: Vector2D) -> Color:
        return self.color


class ImagePigment(Pigment):
    """A textured pigment

    The texture is given through a PFM image."""

    def __init__(self, image: HdrImage):
        self.image = image

    def get_color(self, uv: Vector2D) -> Color:
        col = int(uv.u * self.image.width)
        row = int(uv.v * self.image.height)

        if col >= self.image.width:
            col = self.image.width - 1

        if row >= self.image.height:
            row = self.image.height - 1

        # A nicer solution would implement bilinear interpolation to reduce pixelization artifacts
        # See https://en.wikipedia.org/wiki/Bilinear_interpolation
        return self.image.get_pixel(col, row)


class CheckeredPigment(Pigment):
    """A checkered pigment

    The number of rows/columns in the checkered pattern is tunable, but you cannot have a different number of
    repetitions along the u/v directions."""

    def __init__(self, color1: Color, color2: Color, num_of_steps=10):
        self.color1 = color1
        self.color2 = color2
        self.num_of_steps = num_of_steps

    def get_color(self, uv: Vector2D) -> Color:
        int_u = int(floor(uv.u * self.num_of_steps))
        int_v = int(floor(uv.v * self.num_of_steps))

        return self.color1 if ((int_u % 2) == (int_v % 2)) else self.color2
