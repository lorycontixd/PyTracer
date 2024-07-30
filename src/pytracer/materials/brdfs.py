from .pigments import Pigment, UniformPigment
from pytracer.math.geometry import (
    Vector3D,
    Vector2D,
    Normal,
    Point,
    create_onb_from_z,
    normalized_dot,
)
from pytracer.color import Color, BLACK, WHITE
from pytracer.ray import Ray
from pytracer.pcg import PCG
from math import pi, sqrt, cos, sin, acos, inf


class BRDF:
    """An abstract class representing a Bidirectional Reflectance Distribution Function"""

    def __init__(self, pigment: Pigment = UniformPigment(WHITE)):
        self.pigment = pigment

    def eval(
        self, normal: Normal, in_dir: Vector3D, out_dir: Vector3D, uv: Vector2D
    ) -> Color:
        return BLACK

    def scatter_ray(
        self,
        pcg: PCG,
        incoming_dir: Vector3D,
        interaction_point: Point,
        normal: Normal,
        depth: int,
    ):
        raise NotImplementedError("You cannot call BRDF.scatter_ray directly!")


class DiffuseBRDF(BRDF):
    """A class representing an ideal diffuse BRDF (also called «Lambertian»)"""

    def __init__(self, pigment: Pigment = UniformPigment(WHITE)):
        super().__init__(pigment)

    def eval(
        self, normal: Normal, in_dir: Vector3D, out_dir: Vector3D, uv: Vector2D
    ) -> Color:
        return self.pigment.get_color(uv) * (1.0 / pi)

    def scatter_ray(
        self,
        pcg: PCG,
        incoming_dir: Vector3D,
        interaction_point: Point,
        normal: Normal,
        depth: int,
    ):
        # Cosine-weighted distribution around the z (local) axis
        e1, e2, e3 = create_onb_from_z(normal)
        cos_theta_sq = pcg.random_float()
        cos_theta, sin_theta = sqrt(cos_theta_sq), sqrt(1.0 - cos_theta_sq)
        phi = 2.0 * pi * pcg.random_float()

        return Ray(
            origin=interaction_point,
            dir=e1 * cos(phi) * cos_theta + e2 * sin(phi) * cos_theta + e3 * sin_theta,
            tmin=1.0e-3,
            tmax=inf,
            depth=depth,
        )


class SpecularBRDF(BRDF):
    """A class representing an ideal mirror BRDF"""

    def __init__(
        self, pigment: Pigment = UniformPigment(WHITE), threshold_angle_rad=pi / 1800.0
    ):
        super().__init__(pigment)
        self.threshold_angle_rad = threshold_angle_rad

    def eval(
        self, normal: Normal, in_dir: Vector3D, out_dir: Vector3D, uv: Vector2D
    ) -> Color:
        # We provide this implementation for reference, but we are not going to use it (neither in the
        # path tracer nor in the point-light tracer)
        theta_in = acos(normalized_dot(normal, in_dir))
        theta_out = acos(normalized_dot(normal, out_dir))

        if abs(theta_in - theta_out) < self.threshold_angle_rad:
            return self.pigment.get_color(uv)
        else:
            return Color(0.0, 0.0, 0.0)

    def scatter_ray(
        self,
        pcg: PCG,
        incoming_dir: Vector3D,
        interaction_point: Point,
        normal: Normal,
        depth: int,
    ):
        # There is no need to use the PCG here, as the reflected direction is always completely deterministic
        # for a perfect mirror

        ray_dir = Vector3D(incoming_dir.x, incoming_dir.y, incoming_dir.z).normalize()
        normal = normal.to_vec().normalize()
        dot_prod = normal.dot(ray_dir)

        return Ray(
            origin=interaction_point,
            dir=ray_dir - normal * 2 * dot_prod,
            tmin=1e-5,
            tmax=inf,
            depth=depth,
        )
