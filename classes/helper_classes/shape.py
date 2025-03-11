from shapely import get_num_geometries
from shapely.geometry import LineString, Polygon, MultiPolygon, mapping
from shapely import intersection
from shapely.ops import split

import numpy.linalg as la
import numpy as np

# Represents an origin and vector direction
class Offset:
    def __init__(self, plane_orig, plane_norm):
        self.plane_norm = plane_norm
        self.plane_orig = plane_orig
        self.plane_norm /= la.norm(self.plane_norm)
    
    def __hash__(self):
        to_hash = list(self.plane_norm) + list(self.plane_orig)
        return hash(str(to_hash))
    
    def intersection_line(self, other):
        n1 = np.array(self.plane_norm, dtype=float)
        n2 = np.array(other.plane_norm, dtype=float)
        p1 = np.array(self.plane_orig, dtype=float)
        p2 = np.array(other.plane_orig, dtype=float)

        direction = np.cross(n1, n2)
        if np.allclose(direction, 0):
            return None  # No unique intersection line

        # Find a point on the intersection line
        # We'll solve the system:
        # n1 . p + D1 = 0
        # n2 . p + D2 = 0
        # Where D1 = -n1 . p1 and D2 = -n2 . p2

        D1 = -np.dot(n1, p1)
        D2 = -np.dot(n2, p2)

        # To solve for p, set one coordinate to zero based on the largest component in direction
        abs_direction = np.abs(direction)
        idx = np.argmax(abs_direction)

        # Set the idx-th coordinate to zero and solve for the other two
        # For example, if idx == 0 (x is largest), set x = 0
        # Then solve for y and z
        indices = [0, 1, 2]
        indices.remove(idx)

        A = np.array([
            [n1[indices[0]], n1[indices[1]]],
            [n2[indices[0]], n2[indices[1]]]
        ])
        b = -np.array([D1, D2])

        try:
            solution = np.linalg.solve(A, b)
            point = np.zeros(3)
            point[indices[0]] = solution[0]
            point[indices[1]] = solution[1]
            # The idx-th coordinate remains zero
            return Offset(plane_orig=list(point), plane_norm=list(direction))
        
        except np.linalg.LinAlgError:
            return None  # No unique solution


# stores attributes about a geometric shape
# represents both 3D and 2D representation
# points always assumed to lie on the plane specified by the offset parameter
class Shape:
    def __init__(self, contour: list[list[float]], holes: list[list[list[float]]] = None, offset: Offset = None):
        self.contour: list[list[float]] = contour
        self.holes: list[list[list[float]]] = holes

        self.offset: Offset = offset

        if self.holes is None:
            self.holes = []
        if offset is None:
            self.offset = Offset(plane_norm=[0, 0, 0], plane_orig=[0, 0, 0])
        
    def __hash__(self):
        self.to_2d()
        to_hash = list(self.contour)
        to_hash += list(self.holes)
        to_hash.append(self.offset.__hash__())
        return hash(str(to_hash))

    def get_contour(self):
        return self.contour
    
    def get_holes(self):
        return self.holes
    
    def to_polygon(self):
        self.to_2d()
        return Polygon(self.contour, self.holes)
    
    def to_linestring(self):
        self.to_2d()
        return LineString(self.contour)
    
    def polygon_to_shape(self, polygon: Polygon, offset: Offset):
        exterior_coords = polygon.exterior.coords[:] 
        interior_coords = [interior.coords[:] for interior in polygon.interiors]

        return Shape(contour= exterior_coords, holes = interior_coords, offset=offset)
    
    #slices given shape on given line, returns geometry inside or outside of the accept_shape, as specified by accept_inner
    # if no shape provided, returns biggest shape
    def slice_on_line(self, line, accept_shape = None, accept_inner: bool = None):
        # self.plot_slice(shape = shape + slice, file_name='test.png')
        #print(shape)
        self.to_2d()

        polygon = self.to_polygon()

        slice_ls = LineString(line)
        sliced = split(polygon, slice_ls)

        # print(type(sliced))
        if accept_shape is None:
            biggest_geom = None
            for i in range(get_num_geometries(sliced)):
                cur_geom = sliced.geoms._get_geom_item(i)
                if biggest_geom is None:
                    biggest_geom = cur_geom
                elif cur_geom.area > biggest_geom.area:
                    biggest_geom = cur_geom
            result = self.polygon_to_shape(biggest_geom)
            return result

        else:
            geoms = []
            accept_poly = accept_shape.to_polygon()
            for i in range(get_num_geometries(sliced)):
                cur_geom = sliced.geoms._get_geom_item(i)
                contained: bool = accept_poly.contains(cur_geom)
                if contained == accept_inner:
                    geoms.append(cur_geom)
            geoms = [self.polygon_to_shape(geom) for geom in geoms]
            return geoms

        
    def slice_multiple_on_line(self, shapes, slice, return_all: bool= False):
        results = [self.slice_on_line(shape, slice, return_all=return_all) for shape in shapes]
        return results
    
    
    def get_midpoint_in_slice_along_line(self, line):
        self.to_2d()
        line_ls = LineString(line)

        polygon = self.to_polygon()

        result: LineString = intersection(polygon, line_ls)
        midpoint = result.centroid
        return midpoint
    
    def get_plane_axes(self):
        """
        Given a plane normal, compute two orthonormal vectors lying on the plane.
        
        Args:
            plane_normal (list or array): The normal vector of the plane [nx, ny, nz].
        
        Returns:
            tuple: Two orthonormal vectors (u, v) lying on the plane.
        """
        n = np.array(self.offset.plane_norm, dtype=float)
        n /= np.linalg.norm(n)  # Ensure the normal is a unit vector
        
        # Choose an arbitrary vector not parallel to the normal
        if not np.isclose(n[0], 0) or not np.isclose(n[1], 0):
            arbitrary = np.array([-n[1], n[0], 0])
        else:
            arbitrary = np.array([0, -n[2], n[1]])
        
        u = np.cross(arbitrary, n)
        u /= np.linalg.norm(u)  # Normalize u
        
        v = np.cross(n, u)  # Already orthogonal and normalized
        return u, v

    def points3D_to_2D(self, points3D):
        """
        Converts a set of 3D points to 2D coordinates on the specified plane.

        Args:
            points3D (array-like): Array of 3D points with shape (N, 3).
            plane_origin (array-like): A point on the plane [ox, oy, oz].
            plane_normal (array-like): The normal vector of the plane [nx, ny, nz].

        Returns:
            list: Array of 2D points with shape (N, 2).
        """
        points3D = np.asarray(points3D, dtype=float)
        plane_origin = np.asarray(self.offset.plane_orig, dtype=float)
        u, v = self.get_plane_axes()

        # Compute vectors from plane origin to points
        vectors = points3D - plane_origin  # Shape: (N, 3)

        # Compute 2D coordinates by projecting onto local axes
        x = np.dot(vectors, u)  # Shape: (N,)
        y = np.dot(vectors, v)  # Shape: (N,)

        points2D = np.stack((x, y), axis=-1)  # Shape: (N, 2)
        points2D = [list(p) for p in points2D] #convert back to lists
        return points2D
    
    def points2D_to_3D(self, points2D):
        """
        Converts a set of 2D points on the plane back to 3D coordinates.

        Args:
            points2D (array-like): Array of 2D points with shape (N, 2).
            plane_origin (array-like): A point on the plane [ox, oy, oz].
            plane_normal (array-like): The normal vector of the plane [nx, ny, nz].

        Returns:
            list (float): Array of 3D points with shape (N, 3).
        """
        points2D = np.asarray(points2D, dtype=float)
        plane_origin = np.asarray(self.offset.plane_orig, dtype=float)
        u, v = self.get_plane_axes()

        # Reconstruct 3D points
        points3D = np.outer(points2D[:, 0], u) + np.outer(points2D[:, 1], v) + plane_origin
        points3D = [list(p) for p in points3D]
        return points3D


    def to_2d(self):
        # don't convert if already 2d
        if len(self.contour[0]) == 2:
            return
        
        self.contour = self.points3D_to_2D(self.contour)
        self.holes = [self.points3D_to_2D(hole) for hole in self.holes]
    
    def to_3d(self):
        # don't convert if already 2d
        if len(self.contour[0]) == 3:
            return
        
        self.contour = self.points2D_to_3D(self.contour)
        self.holes = [self.points2D_to_3D(hole) for hole in self.holes]
    

    def contains(self, other):
        other: Shape = other

        p1: Polygon = self.to_polygon()
        p2: Polygon = other.to_polygon()

        return p1.contains(p2)
    
    def get_area(self):
        p: Polygon = self.to_polygon()
        return p.area

    
    def plot_slice(self, file_name = None, fig = None, ax = None):
        import matplotlib.pyplot as plt

        self.to_2d()

        contour = self.contour

        shape_x = [i[0] for i in contour]
        shape_y = [i[1] for i in contour]

        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')

        ax.plot(shape_x, shape_y, 'b-')

        for hole in self.holes:
            hole_x = [i[0] for i in hole]
            hole_y = [i[1] for i in hole]
            plt.plot(hole_x, hole_y, 'r-')

        if not file_name is None:
            fig.savefig(file_name)


# Contains a list of shapes, all with the same offset parameter
class MultiShape:
    def __init__(self, shapes: list[Shape], offset: Offset):
        self.shapes: list[Shape] = shapes
        self.offset: Offset = offset
    
    def __hash__(self):
        to_hash = [s.__hash__() for s in self.shapes]
        to_hash.append(self.offset.__hash__())
        return hash(str(to_hash))

    def to_2d(self):
        for shape in self.shapes:
            shape.to_2d()

    def to_3d(self):
        for shape in self.shapes:
            shape.to_3d()
    
    def to_polygon(self):
        polygons = [shape.to_polygon() for shape in self.shapes]
        return MultiPolygon(polygons)
    
    def polygon_to_shape(self, polygon: MultiPolygon):
        shapes: list[Shape] = []
        convert_shape: Shape = Shape(contour=[], holes=[], offset=self.offset)
        for i in range(get_num_geometries(polygon)):
            cur_geom = polygon.geoms._get_geom_item(i)
            cur_shape = convert_shape.polygon_to_shape(cur_geom, self.offset)
            shapes.append(cur_shape)

        return MultiShape(shapes=shapes, offset=convert_shape.offset)

    def get_midpoint_along_line(self, line):
        self.to_2d()

        convert_shape: Shape = Shape(contour=[], holes=[], offset=self.offset)
        if len(line[0]) > 2:
            line = convert_shape.points3D_to_2D(line)

        line_ls = LineString(line)

        polygon = self.to_polygon()

        result: LineString = intersection(polygon, line_ls)
        midpoint = result.centroid
        if midpoint.is_empty:
            return None

        midpoint = [midpoint.x, midpoint.y]
        midpoint = convert_shape.points2D_to_3D([midpoint])[0]
        return midpoint


    def slice_on_line(self, line: Shape, accept_shape: Shape = None, accept_inner: bool = None):
        # self.plot_slice(shape = shape + slice, file_name='test.png')
        #print(shape)
        self.to_2d()

        polygon = self.to_polygon()
        slice_ls: LineString = line.to_linestring()
        sliced = split(polygon, slice_ls)

        #print(sliced, '/n')

        # print(type(sliced))
        if accept_shape is None:
            biggest_geom: Polygon = None
            for i in range(get_num_geometries(sliced)):
                cur_geom: Polygon = sliced.geoms._get_geom_item(i)
                if biggest_geom is None:
                    biggest_geom = cur_geom
                elif cur_geom.area > biggest_geom.area:
                    biggest_geom = cur_geom
            #print(type(biggest_geom))
            biggest_geom = MultiPolygon([biggest_geom])
            result = self.polygon_to_shape(biggest_geom)
            return result

        else:
            geoms = []
            accept_poly = accept_shape.to_polygon()
            for i in range(get_num_geometries(sliced)):
                cur_geom: Polygon = sliced.geoms._get_geom_item(i)

                if not cur_geom.is_valid:
                    continue
                try:
                    contained: bool = accept_poly.contains(cur_geom)
                    if contained == accept_inner:
                        geoms.append(cur_geom)
                except Exception as e:
                    print(e)
            result_poly = MultiPolygon(geoms)
            shapes: MultiShape = self.polygon_to_shape(result_poly)
            return shapes
        
    def plot_slice(self, file_name = None, fig = None, ax = None):
        import matplotlib.pyplot as plt

        self.to_2d()

        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
        
        for shape in self.shapes:
            shape.plot_slice(fig=fig, ax=ax)

        if not file_name is None:
            fig.savefig(file_name)

