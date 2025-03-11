import math
import numpy as np

#used only for plotting function
from classes.helper_classes.shape import Shape, Offset, MultiShape

from classes.helper_classes.model import Model


def create_slots_from_midpoints(midpoints: list[Offset], magnitude: float, orth_magnitude: float, slice_offset: Offset, dir: int):
    slots: list[Shape] = []
    for midpoint in midpoints:
        if midpoint is None:
            continue

        cur_slot = create_slot_from_midpoint(
        midpoint=midpoint,
        magnitude=magnitude,
        orth_magnitude=orth_magnitude,
        slice_offset=slice_offset,
        dir=dir)

        slots.append(cur_slot)
    return slots


#same as create_slots_from_midpoints, but uses distance to origin of slice as magnitude for slots
def create_radial_slots_from_midpoints(midpoints: list[Offset], orth_magnitude: float, slice_offset: Offset, dir: int):
    slots: list[Shape] = []
    for midpoint in midpoints:
        if midpoint is None:
            continue

        center = np.asarray(slice_offset.plane_orig)
        mid_pos = np.asarray(midpoint.plane_orig)
        magnitude = np.linalg.norm(mid_pos - center)

        cur_slot = create_slot_from_midpoint(midpoint=midpoint,
                                            magnitude=magnitude,
                                            orth_magnitude= orth_magnitude,
                                            slice_offset=slice_offset,
                                            dir=dir)

        slots.append(cur_slot)

    return slots

def create_slot_from_midpoint(midpoint: Offset, magnitude: float, orth_magnitude: float, slice_offset: Offset, dir: int):
    center_offset_magnitude = orth_magnitude

    midpoint_dir = np.asarray(midpoint.plane_norm)
    midpoint_pos = np.asarray(midpoint.plane_orig)
    slice_norm = np.asarray(slice_offset.plane_norm)
    slice_pos = np.asarray(slice_offset.plane_orig)

    orth_dir = np.cross(midpoint_dir, slice_norm)
    center_offset_pos = orth_dir * center_offset_magnitude

    depth_offset_pos = midpoint_dir * dir * magnitude

    slot: list = []

    slot += [list(midpoint_pos - center_offset_pos + depth_offset_pos + slice_pos)]
    slot += [list(midpoint_pos - center_offset_pos + slice_pos)]
    slot += [list(midpoint_pos + center_offset_pos + slice_pos)]
    slot += [list(midpoint_pos + center_offset_pos + depth_offset_pos + slice_pos)]
    slot.append(slot[0])

    slot: Shape = Shape(contour=slot, offset=slice_offset)

    return slot



class ModelSegmental:
    def __init__(self, source: str, params: dict) -> None:
        print("Initializing model:", source)
        self.model = Model(source)
        self.params = params
        self.segmental_obj = None

        #initialize correct image segmental object
        if self.params['type'] == "grid_slice":
            self.segmental_obj = GridSliceSegmental(self.model, self.params)
        elif self.params['type'] == "radial_slice":
            self.segmental_obj = RadialSliceSegmental(self.model, self.params)
        else:
            raise ValueError("Invalid image segmental type.")
    
    def create_segmental(self) -> dict:
        return self.segmental_obj.create_segmental()


class GridSliceSegmental:
    def __init__(self, model: Model, params: dict) -> None:
        self.model: Model = model
        self.params: dict = params

        self.model.scale_to_height(params['height'])
    
    def get_slices(self) -> dict:
        bounds = self.model.bounds

        #generate length slices, length is x, slices along length
        #length is horizontal pieces
        z_min, z_max = bounds[2][0], bounds[2][1]

        #handle vertical offset
        if self.params['vertical_offset'] < 0:
            z_min -= self.params['vertical_offset']
        else:
            z_max += self.params['vertical_offset']


        n_length_slices = self.params['length_slices']
        length_slices_offset = (z_max - z_min) / (n_length_slices + 1)
        length_offsets: list[float] = [z_min + i*length_slices_offset for i in range(1, n_length_slices+1)]
        length_offsets: list[Offset] = [Offset(plane_norm=[0, 0, -1], plane_orig = [0, 0, o]) for o in length_offsets]
        length_slices: list = []
        for cur_length_offset in length_offsets:
            length_slice: MultiShape = self.model.slice_on_plane(cur_length_offset)
            length_slices.append(length_slice)

        #generate height slices, height is z, slices along height
        #height is vertical pieces
        x_min, x_max = bounds[0][0], bounds[0][1]
        n_height_slices = self.params['height_slices']
        height_slices_offset = (x_max - x_min) / (n_height_slices + 1)
        height_offsets: list[float] = [x_min + i*height_slices_offset for i in range(1, n_height_slices+1)]
        height_offsets: list[Offset] = [Offset(plane_norm=[1, 0, 0], plane_orig = [o, 0, 0]) for o in height_offsets]
        height_slices: list = []
        for cur_height_offset in height_offsets:
            height_slice: MultiShape = self.model.slice_on_plane(cur_height_offset)
            height_slices.append(height_slice)


        length_dict = {'offsets': length_offsets,
                       'slices': length_slices}
        height_dict = {'offsets': height_offsets,
                       'slices': height_slices}
        slices = {'length': length_dict,
                  'height': height_dict}
        
        return slices

    # Calculate midpoints on each slice where slots will be
    # Store midpoints as offsets, with the direction indicating direction of slot
    def get_slot_data(self, slices: dict):
        max_magnitude = self.model.get_max_magnitude()

        #generate length slices slot data
        length_slices: list[MultiShape] = slices['length']['slices']
        height_slices: list[MultiShape] = slices['height']['slices']

        midpoints_dict: dict = {}

        for slice in length_slices + height_slices:
            midpoints_dict[slice] = []

        for length_slice in length_slices:
            length_offset: Offset = length_slice.offset

            for height_slice in height_slices:
                height_offset: Offset = height_slice.offset
                line = length_offset.intersection_line(height_offset)

                line_offset = np.asarray(line.plane_norm) * max_magnitude
                line_origin = np.asarray(line.plane_orig)

                line_p1 = list(line_origin + line_offset)
                line_p2 = list(line_origin - line_offset)

                midpoint: list[float] = length_slice.get_midpoint_along_line([line_p1, line_p2])
                #print(midpoint)

                midpoint_offset: Offset = Offset(plane_orig=midpoint, plane_norm=line.plane_norm)

                if midpoint is None:
                    midpoints_dict[length_slice].append(None)
                    midpoints_dict[height_slice].append(None)
                else:
                    midpoints_dict[length_slice].append(midpoint_offset)
                    midpoints_dict[height_slice].append(midpoint_offset)

        return midpoints_dict


    def get_slots(self, slot_data: dict, slices: dict):
        max_magnitude = self.model.get_max_magnitude()
        kerf = self.params['kerf']
        center_offset_magnitude = (self.params['thickness'] - kerf) / 2

        slots_dict: dict = {}

        for length_slice in slices['length']['slices']:
            midpoints = slot_data[length_slice]
            slice_offset: Offset = length_slice.offset
            slots = create_slots_from_midpoints(midpoints=midpoints,
                                                     magnitude=max_magnitude,
                                                     orth_magnitude = center_offset_magnitude,
                                                     slice_offset=slice_offset,
                                                     dir=1)
            slots_dict[length_slice] = slots
        
        #generate height slots
        for height_slice in slices['height']['slices']:
            midpoints = slot_data[height_slice]
            slice_offset: Offset = height_slice.offset
            slots = create_slots_from_midpoints(midpoints=midpoints,
                                                     magnitude=max_magnitude,
                                                     orth_magnitude = center_offset_magnitude,
                                                     slice_offset=slice_offset,
                                                     dir=-1)
            slots_dict[height_slice] = slots
        
        return slots_dict


    def create_segmental(self):
        slices: dict = self.get_slices()

        slot_data: dict = self.get_slot_data(slices)

        slots_and_contours: dict = self.get_slots(slot_data, slices)

        print("Cutting slots")

        pieces: list[Shape] = []
        for slice, slots in zip(slots_and_contours.keys(), slots_and_contours.values()):
            #redeclare for type hints
            slice: MultiShape = slice
            slots: list[Shape] = slots

            result: MultiShape = slice        
            for slot in slots:
                result = result.slice_on_line(line=slot,
                                    accept_shape=slot,
                                    accept_inner=False)

            pieces += result.shapes
    
        return pieces








class RadialSliceSegmental:
    def __init__(self, model: Model, params: dict) -> None:
        self.model: Model = model
        self.params: dict = params

        self.model.scale_to_height(params['height'])
    
    def get_slices(self) -> dict:
        bounds = self.model.bounds

        #generate length slices, length is x, slices along length
        #length is horizontal pieces
        z_min, z_max = bounds[2][0], bounds[2][1]

        #handle vertical offset
        if self.params['vertical_offset'] < 0:
            z_min -= self.params['vertical_offset']
        else:
            z_max += self.params['vertical_offset']

        
        n_length_slices = self.params['length_slices']
        length_slices_offset = (z_max - z_min) / (n_length_slices + 1)
        length_offsets: list[float] = [z_min + i*length_slices_offset for i in range(1, n_length_slices+1)]
        length_offsets: list[Offset] = [Offset(plane_norm=[0, 0, -1], plane_orig = [0, 0, o]) for o in length_offsets]
        length_slices: list[MultiShape] = []
        for cur_length_offset in length_offsets:
            length_slice: MultiShape = self.model.slice_on_plane(cur_length_offset)
            length_slices.append(length_slice)

        #generate radial slices, height is z, slices along height
        #height is vertical pieces
        #x_min, x_max = bounds[0][0], bounds[0][1]
        n_radial_slices = self.params['radial_slices']
        radial_slices_offset = 2*math.pi / (n_radial_slices + 0)
        radial_offsets: list[float] = [i*radial_slices_offset for i in range(1, n_radial_slices+1)]
        radial_offsets: list[Offset] = [Offset(plane_orig=[0, 0, 0],
                                               plane_norm=[-math.cos(cur_radial_offset), -math.sin(cur_radial_offset), 0])
                                               for cur_radial_offset in radial_offsets]
        radial_slices: list[MultiShape] = []
        for cur_radial_offset in radial_offsets:
            radial_slice: MultiShape = self.model.slice_on_plane(cur_radial_offset)
            radial_slices.append(radial_slice)

        slices = {'length': length_slices,
                  'radial': radial_slices}
        
        return slices


    # Slices slice with clearance and gets correct side
    def process_radial_slice(self, radial_slice: MultiShape, line: Offset):
        clearance = self.params['thickness']
        clearance_offset = np.asarray(line.plane_norm)*clearance

        line_dir = np.asarray(line.plane_norm)
        line_pos = np.asarray(line.plane_orig)

        midpoint_pos = clearance_offset+line_pos
        midpoint_offset: Offset = Offset(plane_orig=list(midpoint_pos), plane_norm=list(line_dir))

        magnitude = self.model.get_max_magnitude()

        cut_shape = create_slot_from_midpoint(midpoint=midpoint_offset,
                                                   magnitude=magnitude,
                                                   orth_magnitude=magnitude,
                                                   slice_offset=radial_slice.offset,
                                                   dir=1)
        
        result_slice = radial_slice.slice_on_line(line=cut_shape,
                                                  accept_shape=cut_shape,
                                                  accept_inner=True)

        return result_slice


    def process_radial_slices(self, slices: list[MultiShape], perp_offset: Offset):
        result_slices: list[MultiShape] = []
        for slice in slices:
            line = perp_offset.intersection_line(slice.offset)
            result_slice = self.process_radial_slice(radial_slice=slice, line=line)
            result_slices.append(result_slice)
        return result_slices


    def get_slot_data(self, slices: dict):
        max_magnitude = self.model.get_max_magnitude()

        length_slices: list[MultiShape] = slices['length']
        radial_slices: list[MultiShape] = slices['radial']

        midpoints_dict: dict = {}
        for slice in length_slices+radial_slices:
            midpoints_dict[slice] = []

        for length_slice in length_slices:
            length_offset: Offset = length_slice.offset
            for radial_slice in radial_slices:
                radial_offset: Offset = radial_slice.offset

                line = length_offset.intersection_line(radial_offset)

                line_offset = np.asarray(line.plane_norm) * max_magnitude
                line_origin = np.asarray(line.plane_orig)
                #print(line_origin)

                line_p1 = list(line_origin + line_offset)
                line_p2 = list(line_origin)

                midpoint: list[float] = length_slice.get_midpoint_along_line([line_p1, line_p2])
                #print(midpoint)

                midpoint_offset: Offset = Offset(plane_orig=midpoint, plane_norm=line.plane_norm)

                if midpoint is None:
                    midpoints_dict[length_slice].append(None)
                    midpoints_dict[radial_slice].append(None)
                else:
                    midpoints_dict[length_slice].append(midpoint_offset)
                    midpoints_dict[radial_slice].append(midpoint_offset)

        return midpoints_dict


    def get_slots(self, slot_data: dict, slices: dict):
        max_magnitude = self.model.get_max_magnitude()
        kerf = self.params['kerf']
        center_offset_magnitude = (self.params['thickness'] - kerf) / 2

        slots_dict: dict = {}

        for length_slice in slices['length']:
            midpoints = slot_data[length_slice]
            slice_offset: Offset = length_slice.offset
            slots = create_radial_slots_from_midpoints(midpoints=midpoints,
                                                     orth_magnitude= center_offset_magnitude,
                                                     slice_offset=slice_offset,
                                                     dir=1)
            slots_dict[length_slice] = slots
        
        #generate height slots
        for radial_slice in slices['radial']:
            midpoints = slot_data[radial_slice]
            slice_offset: Offset = radial_slice.offset
            slots = create_slots_from_midpoints(midpoints=midpoints,
                                                     magnitude=max_magnitude,
                                                     orth_magnitude=center_offset_magnitude,
                                                     slice_offset=slice_offset,
                                                     dir=-1)
            slots_dict[radial_slice] = slots
        
        return slots_dict


    def create_segmental(self):
        print("Slicing on planes")
        slices: dict = self.get_slices()

        print("Processing radial slices")
        perp_offset: Offset = slices['length'][0].offset
        slices['radial'] = self.process_radial_slices(slices['radial'], perp_offset)


        print("Building slots")
        slot_data: dict = self.get_slot_data(slices)

        slots_and_contours: dict = self.get_slots(slot_data, slices)

        print("Cutting slots")

        pieces: list[Shape] = []
        for slice, slots in zip(slots_and_contours.keys(), slots_and_contours.values()):
            #redeclare for type hints
            slice: MultiShape = slice
            slots: list[Shape] = slots

            result: MultiShape = slice 
            for slot in slots:
                result = result.slice_on_line(line=slot,
                                    #accept_shape=slot,
                                    #accept_inner=False
                                    )

            pieces += result.shapes

        return pieces






