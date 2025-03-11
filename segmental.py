import json
import os

from classes.segmental_classes.model_segmental import ModelSegmental
from classes.helper_classes.shape import Shape
from classes.helper_classes.model_export import ModelExport
from classes.helper_classes.vector_export import VectorExport

class SegmentalMaker:
    def __init__(self, file: str = None) -> None:
        self.input = None
        self.segmental = None
        self.export = None
        if file != None:
            self.load(file)
        
    def load(self, file: str):
        print("Initializing segmental maker")
        with open(file, 'r') as file:
            self.input = json.load(file)
            #print(self.input)
        
        type = self.input['type']
        if type == "3d":
            self.segmental = ModelSegmental(self.input['source'], self.input['params'])
        else:
            raise ValueError("Segmental type invalid.")
        
        export_type = self.input['export']
        if export_type == "stl":
            self.export = ModelExport(self.input['extrude_height'])
        elif export_type == "svg":
            print("Export type SVG")
            self.export = VectorExport()
        else:
            raise ValueError("Segmental export type invalid.")
        print("Initialized segmental maker")
        


    #creates correct segmental object and saves exported files to folder
    def make_segmental(self):
        print("Started creating segmental")
        if self.input == None:
            raise ValueError('No loaded input file.')

        pieces = self.segmental.create_segmental()

        #try making folder directory in case it does not exist
        try:
            os.makedirs((self.input['export_folder']))
        except:
            pass
        
        file_name = f"{self.input['export_prefix']}"
        output_path = os.path.join(self.input['export_folder'], file_name)
        print("Exporting segmental")
        if type(self.export) is ModelExport:
            print("Exporting 3D models")
            self.export.export(shapes = pieces, file_name = output_path)
        elif type(self.export) is VectorExport:
            print("Exporting Vector files")
            self.export.export(shapes = pieces, file_name = output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python segmental.py <input_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    test = SegmentalMaker(input_file)
    test.make_segmental()

