from pytracer.math.geometry import Point, Vector3D
from pytracer.ray import Ray


class AABB:
    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 0:
            self.min = Point(0, 0, 0)
            self.max = Point(0, 0, 0)
        elif len(args) == 1:
            if not isinstance(args[0], Point):
                raise TypeError("AABB constructor expects at least a Point.")
            self.min = args[0]
            self.max = args[0]
        elif len(args) == 2:
            if not isinstance(args[0], Point) or not isinstance(args[1], Point):
                raise TypeError("AABB constructor expects two Points.")
            self.min = args[0]
            self.max = args[1]
        else:
            raise TypeError("AABB constructor expects at most two Points.")

    def __str__(self) -> str:
        return f"AABB(min={self.min}, max={self.max})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, AABB):
            return False
        return self.min == other.min and self.max == other.max

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __getitem__(self, item):
        if item == 0:
            return self.min
        elif item == 1:
            return self.max
        else:
            raise IndexError("Index out of range")

    def corner(self, corner: int) -> Point:
        """Return the `corner`-th corner of the AABB"""
        if corner < 0 or corner > 7:
            raise ValueError("The corner index must be between 0 and 7")
        return Point(
            x=self.min.x if corner & 1 == 0 else self.max.x,
            y=self.min.y if corner & 2 == 0 else self.max.y,
            z=self.min.z if corner & 4 == 0 else self.max.z,
        )

    def diagonal(self, normalize: bool = False) -> Vector3D:
        """Return the diagonal of the AABB"""
        diag = (self.max - self.min).to_vector()
        return diag.normalize() if normalize else diag

    def expand(self, scale: float) -> None:
        """Expand the AABB by a factor `scale`.
        This means that the AABB will grow by `scale` times its diagonal.
        """
        diag = self.diagonal()
        self.min -= diag * scale
        self.max += diag * scale

    def contains(self, point: Point) -> bool:
        """Check if the AABB contains the point"""
        return all(self.min[i] <= point[i] <= self.max[i] for i in range(3))

    def intersection(self, other: "AABB") -> "AABB":
        """Return the intersection of two AABBs"""
        return AABB(
            Point(
                max(self.min.x, other.min.x),
                max(self.min.y, other.min.y),
                max(self.min.z, other.min.z),
            ),
            Point(
                min(self.max.x, other.max.x),
                min(self.max.y, other.max.y),
                min(self.max.z, other.max.z),
            ),
        )

    def lerp(self, point: Point) -> Point:
        """Linearly interpolate the point inside the AABB"""
        return Point(
            x=(point.x - self.min.x) / (self.max.x - self.min.x),
            y=(point.y - self.min.y) / (self.max.y - self.min.y),
            z=(point.z - self.min.z) / (self.max.z - self.min.z),
        )

    def overlaps(self, other: "AABB") -> bool:
        """Check if the AABB overlaps with another AABB.
        Function useful to detect physical collisions.
        """
        return all(
            self.min[i] < other.max[i] and self.max[i] > other.min[i] for i in range(3)
        )

    def overlaps_delta(self, other: "AABB", delta: float) -> bool:
        """Check if the AABB overlaps with another AABB, with a delta.
        Function useful to detect physical collisions.
        """
        return all(
            self.min[i] - delta < other.max[i] + delta
            and self.max[i] + delta > other.min[i] - delta
            for i in range(3)
        )

    def ray_intersects(self, ray: Ray) -> bool:
        """Check if the AABB intersects with a ray"""
        tmin = -float("inf")
        tmax = float("inf")
        for i in range(3):
            if ray.direction[i] == 0:
                if ray.origin[i] < self.min[i] or ray.origin[i] > self.max[i]:
                    return False
            else:
                t1 = (self.min[i] - ray.origin[i]) / ray.direction[i]
                t2 = (self.max[i] - ray.origin[i]) / ray.direction[i]
                tmin = max(tmin, min(t1, t2))
                tmax = min(tmax, max(t1, t2))
                if tmin > tmax:
                    return False
        return True

    def surface_area(self) -> float:
        """Return the surface area of the AABB"""
        diag = self.diagonal()
        return 2 * (diag.x * diag.y + diag.x * diag.z + diag.y * diag.z)

    def volume(self) -> float:
        """Return the volume of the AABB"""
        diag = self.diagonal()
        return diag.x * diag.y * diag.z

    def union(self, other: "AABB") -> "AABB":
        """Return the union of two AABBs"""
        return AABB(
            Point(
                min(self.min.x, other.min.x),
                min(self.min.y, other.min.y),
                min(self.min.z, other.min.z),
            ),
            Point(
                max(self.max.x, other.max.x),
                max(self.max.y, other.max.y),
                max(self.max.z, other.max.z),
            ),
        )

    def create_sphere(self) -> ("Point", float):
        """Create a sphere that contains the AABB"""
        center = (self.min + self.max) / 2
        radius = (self.max - center).norm()
        return center, radius
