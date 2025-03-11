import triangle
from stl import mesh
import numpy as np
import copy

from classes.helper_classes.shape import Shape, Offset, MultiShape

#reverse list
def r(l):
    return l [::-1]

class ModelExport:
    def __init__(self, extrude_height) -> None:
        self.height = extrude_height


    # Extrude 2D shape into 3D mesh
    # Handles concave triangulation and holes
    def extrude(self, shape):
        """
        Extrude a 2D shape (with optional holes) by `self.height`.

        shape.contour is the outer boundary (closed ring).
        shape.holes is a list of one or more closed rings for holes.
        """
        # Ensure shape is in 2D
        shape.to_2d()
        
        # Outer boundary (remove the last duplicate point if closed)
        outer_points = shape.contour[:-1]
        
        # Holes (each hole is also closed; remove duplicate last point)
        hole_rings = [hole[:-1] for hole in shape.holes]

        # -----------------------------------------------------------
        # 1. Build up PSLG for triangle
        # -----------------------------------------------------------
        #
        # We need:
        #   poly['vertices'] : Nx2 array of all vertex coordinates
        #   poly['segments'] : Mx2 array of edges for boundary + holes
        #   poly['holes']    : Kx2 array of points strictly inside each hole
        #
        all_points = list(outer_points)
        segments   = []
        
        # Add segments for the outer boundary
        outer_count = len(outer_points)
        for i in range(outer_count):
            segments.append([i, (i+1) % outer_count])
        
        # Keep track of how many vertices so far, so we know where hole vertices begin
        current_index = outer_count
        
        holes_centers = []
        
        # Add segments for each hole boundary
        for hole_ring in hole_rings:
            n = len(hole_ring)
            start_idx = current_index
            
            # Add the hole's vertices
            all_points.extend(hole_ring)
            
            # Add the hole's boundary edges
            for i in range(n):
                segments.append([start_idx + i, start_idx + (i+1) % n])
            
            current_index += n
            
            # We must provide one interior (seed) point per hole in poly['holes'].
            # A common quick approach is the average of the hole boundary points
            # which hopefully lies in the interior if the hole is simple.
            c = np.mean(hole_ring, axis=0)
            holes_centers.append(c)
        
        poly = {
            'vertices':  np.array(all_points, dtype=np.float64),
            'segments':  np.array(segments,   dtype=np.int32)
        }
        
        if holes_centers:
            poly['holes'] = np.array(holes_centers, dtype=np.float64)

        # -----------------------------------------------------------
        # 2. Triangulate
        # -----------------------------------------------------------
        # The 'p' switch tells triangle to use the PSLG we provide 
        # (i.e. respect boundary segments, preserve input vertices).
        # You can add more switches as needed (e.g. "pq" to ensure
        # quality constraints).
        triangulated = triangle.triangulate(poly, 'p')
        
        vertices_2d = triangulated['vertices']       # Nx2
        faces_2d    = triangulated['triangles']      # Mx3

        # -----------------------------------------------------------
        # 3. Build 3D vertices for top and bottom
        # -----------------------------------------------------------
        # Bottom: z=0, Top: z=height
        bottom_3d = np.hstack((vertices_2d, np.zeros((len(vertices_2d), 1))))
        top_3d    = bottom_3d + np.array([0, 0, self.height])
        
        # Combine
        all_3d_vertices = np.vstack((bottom_3d, top_3d))
        num_base_vertices = len(bottom_3d)   # N

        # -----------------------------------------------------------
        # 4. Create faces for top and bottom
        # -----------------------------------------------------------
        # faces_2d references the bottom_3d by index [0..N-1].
        # The top face is the same faces, offset by + N.
        #
        # Reverse the bottom faces to flip normals downward:
        bottom_faces = [r(face) for face in faces_2d]
        
        # The top face uses the same faces, offset by +num_base_vertices
        top_faces = faces_2d + num_base_vertices
        
        # -----------------------------------------------------------
        # 5. Create side walls for both the outer boundary and each hole
        # -----------------------------------------------------------
        # We can loop over each "ring" of the original boundary (outer + holes),
        # and extrude it by connecting bottom[i]->bottom[i+1]->top[i+1]->top[i].
        # Because 'p' switch in triangle usually preserves the input vertices
        # at their same indices in the output array, we can rely on them being 
        # in the same order in the first part of `vertices_2d`.
        
        # For clarity, let's define a helper to extrude one ring:
        def build_side_faces_ring(start_idx, count):
            ring_side_faces = []
            for i in range(count):
                i_next = (i + 1) % count
                b0 = start_idx + i       # bottom index of current
                b1 = start_idx + i_next  # bottom index of next
                t0 = b0 + num_base_vertices
                t1 = b1 + num_base_vertices
                # Two triangles for the quad side:
                # (b0, b1, t1), (b0, t1, t0)
                ring_side_faces.append([b0, b1, t1])
                ring_side_faces.append([b0, t1, t0])
            ring_side_faces = [r(face) for face in ring_side_faces]
            return ring_side_faces

        side_faces = []
        
        # Outer boundary side walls
        outer_side = build_side_faces_ring(0, outer_count)
        side_faces.extend(outer_side)
        
        # Each hole boundary side walls
        hole_start = outer_count
        for hole_ring in hole_rings:
            ring_len = len(hole_ring)
            hole_side = build_side_faces_ring(hole_start, ring_len)
            side_faces.extend(hole_side)
            hole_start += ring_len
        
        # -----------------------------------------------------------
        # 6. Combine all faces
        # -----------------------------------------------------------
        # Convert lists to numpy arrays for constructing the STL mesh
        bottom_faces = np.array(bottom_faces, dtype=np.int32)
        top_faces    = np.array(top_faces,    dtype=np.int32)
        side_faces   = np.array(side_faces,   dtype=np.int32)

        all_faces = np.vstack((bottom_faces, top_faces, side_faces))

        # -----------------------------------------------------------
        # 7. Build the final 3D mesh
        # -----------------------------------------------------------
        model_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
        for i, face in enumerate(all_faces):
            for j in range(3):
                model_mesh.vectors[i][j] = all_3d_vertices[face[j], :]

        return model_mesh


    #rotates mesh given rotation matrix
    def rotate_mesh(self, mesh: mesh.Mesh, rotation):
        mesh.vectors = np.dot(mesh.vectors, rotation.T)
        return mesh

    def translate_mesh(self, mesh: mesh.Mesh, translation_vector):
        mesh.vectors += translation_vector
        return mesh
    
    def combine_meshes(self, meshes):
        combined_mesh = mesh.Mesh(np.concatenate([m.data.copy() for m in meshes]))
        return combined_mesh
    
    # rotates mesh to be along a plane, used for 3d reconstruction of slices
    def align_mesh_with_plane(self, mesh: mesh.Mesh, plane: Offset, alignment_translation: list):
        convert_shape: Shape = Shape(contour=[], offset=plane)

        for face_idx in range(len(mesh.vectors)):
            #final_verts: list = []

            for vertex_idx in range(len(mesh.vectors[face_idx])):
                cur_vertex = mesh.vectors[face_idx, vertex_idx]
                cur_vertex += alignment_translation

                cur_vertex_z = cur_vertex[2]
                cur_vertex_xy = cur_vertex[0:2]
                converted_xy = np.asarray(convert_shape.points2D_to_3D([cur_vertex_xy])[0])

                norm_vec = np.asarray(plane.plane_norm)
                final_vertex = converted_xy + norm_vec*cur_vertex_z

                mesh.vectors[face_idx, vertex_idx] = final_vertex


    def export_combined_mesh(self, mesh_dict: dict, file_name: str):
        #combine meshes
        final_meshes = []
        #rotate and translate length meshes
        for shape, cur_mesh in zip(mesh_dict.keys(), mesh_dict.values()):
            new_mesh = copy.deepcopy(cur_mesh)
            new_mesh_offset = shape.offset

            #center on extrude height so stuff lines up
            alignment_translation = np.asarray([0, 0, -1*self.height/2])

            self.align_mesh_with_plane(new_mesh, new_mesh_offset, alignment_translation)
            final_meshes.append(new_mesh)

        final_mesh = self.combine_meshes(final_meshes)
        final_mesh.save(f"{file_name}_combined.stl")

        
    def export(self, shapes: list[Shape], file_name: str):
        counter = 1
        mesh_dict: dict = {}
        for shape in shapes:
            model_mesh = self.extrude(shape)
            model_mesh.save(f"{file_name}{counter}.stl")
            mesh_dict[shape] = model_mesh
            print(f"Exported {file_name}{counter}.stl")
            counter += 1
        self.export_combined_mesh(mesh_dict, file_name)
        print("Done!")


