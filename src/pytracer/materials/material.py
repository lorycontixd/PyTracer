from dataclasses import dataclass
from .brdfs import BRDF, DiffuseBRDF
from .pigments import Pigment, UniformPigment
from pytracer.color import BLACK


@dataclass
class Material:
    """A material"""

    brdf: BRDF = DiffuseBRDF()
    emitted_radiance: Pigment = UniformPigment(BLACK)
