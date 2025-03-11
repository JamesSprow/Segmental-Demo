import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl.mesh import Mesh

from classes.helper_classes.meshcut import merge_close_vertices, TriangleMesh, Plane, cross_section_mesh
from classes.helper_classes.shape import Shape, Offset, MultiShape


class Model:
    #loads and initializes an stl model
    def __init__(self, file: str) -> None:
        self.verts, self.faces = self.load_stl(file)
        #print(self.verts, '\n\n', self.faces)
        self.mesh = TriangleMesh(self.verts, self.faces)
        self.bounds = self.get_bounds()
    
    #imports an stl mesh given file name
    def load_stl(self, stl_fname):
        m = Mesh.from_file(stl_fname)

        # Flatten our vert array to Nx3 and generate corresponding faces array
        verts = m.vectors.reshape(-1, 3)
        faces = np.arange(len(verts)).reshape(-1, 3)

        verts, faces = merge_close_vertices(verts, faces)
        return verts, faces
    
    # detects if some shapes are completely enclosed by other shapes
    # if so, combines them into one shape with holes
    def detect_holes(self, shapes: list[Shape]) -> list[Shape]:
        sorted_shapes: list[Shape] = [shape for shape in shapes]
        result_shapes: list[Shape] = []

        if len(shapes) <= 1:
            return sorted_shapes

        key_fn = lambda shape: shape.get_area()
        sorted(sorted_shapes, key=key_fn)

        # detect holes and remove shapes that were holes
        removed_shapes: list[int] = []
        for i in range(len(sorted_shapes)):
            if i in removed_shapes:
                continue

            cur_contour_shape: Shape = Shape(contour=sorted_shapes[i].contour, offset=sorted_shapes[i].offset)
            holes: list = []
            for j in range(i+1, len(sorted_shapes)):
                if j in removed_shapes:
                    continue

                cur_hole_shape: Shape = sorted_shapes[j]
                if cur_contour_shape.contains(cur_hole_shape):
                    holes.append(cur_hole_shape.contour)
                    removed_shapes.append(j)

            cur_contour_shape.holes = holes
            result_shapes.append(cur_contour_shape)

        return result_shapes



    #slices mesh on a plane
    def slice_on_plane(self, slice_offset: Offset) -> MultiShape:
        plane = Plane(slice_offset.plane_orig, slice_offset.plane_norm)
        cross_sections = cross_section_mesh(self.mesh, plane)
        cross_section_shapes: list[Shape] = []

        for cross_section in cross_sections:
            cross_section_shape = Shape(contour=[list(p) for p in cross_section],
                                        holes= None,
                                        offset = slice_offset)
            cross_section_shapes.append(cross_section_shape)
        
        # detect and combine shapes if some are contained within each other
        cross_section_shapes = self.detect_holes(cross_section_shapes)
        
        result = MultiShape(cross_section_shapes, slice_offset)

        return result
    
    #get bounding box for all points in model, used to determine slice positions later
    def get_bounds(self):
        min_x = min([point[0] for point in self.verts])
        max_x = max([point[0] for point in self.verts])
        min_y = min([point[1] for point in self.verts])
        max_y = max([point[1] for point in self.verts])
        min_z = min([point[2] for point in self.verts])
        max_z = max([point[2] for point in self.verts])
        bounds = [[min_x, max_x],
                  [min_y, max_y],
                  [min_z, max_z]]
        return bounds
    
    # Rough approximation of distance of furthest point from origin in the model, overestimates always
    def get_max_magnitude(self):
        bounds = self.bounds
        y_min, y_max = bounds[1][0], bounds[1][1]
        z_min, z_max = bounds[2][0], bounds[2][1]
        x_min, x_max = bounds[0][0], bounds[0][1]
        max_bound = 2*max(np.abs([y_max, y_min, z_max, z_min, x_min, x_max]))
        max_magnitude = np.sqrt(3*max_bound**2)
        return max_magnitude
    
    # height is the z coordinate
    # scales model to fit user provided height
    def scale_to_height(self, height: float):
        model_height = (self.bounds[2][0] - self.bounds[2][1])
        scale_factor = height/model_height
        self.verts = [[coordinate*scale_factor for coordinate in point] for point in self.verts]
        self.mesh = TriangleMesh(self.verts, self.faces)
        self.bounds = self.get_bounds()
    
    # plots list of shapes using matplotlib, used for debugging
    def plot_slices(self, slices: list, name: str = 'test.png'):
         # Create a new figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        # Plot the line

        for slice in slices:
            #print(slice)
            sliceArray = np.asarray(slice)
            x = sliceArray[:, 0]
            y = sliceArray[:, 1]
            z = sliceArray[:, 2]

            ax.plot(x, y, z, 'b-', alpha = .5)    

        fig.savefig(name)
        