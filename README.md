# README
================

## Segmental Program
-----------------

### Overview

The Segmental program is a work-in-progress tool designed to segment 3D models into individual pieces. The program uses a combination of geometric algorithms and mesh manipulation techniques to create segmental models from input 3D files.

### Structure

The program is structured into several modules:

* `segmental.py`: The main entry point of the program, responsible for loading input files, creating segmental models, and exporting the results.
* `classes/segmental_classes/`: Contains the core classes for segmental model creation, including `ModelSegmental`, `RadialSliceSegmental`, and `GridSliceSegmental`.
* `classes/helper_classes/`: Provides utility classes for mesh manipulation, shape creation, and export functions.

### Running the Program

To run the program, execute the `segmental.py` file with a valid input JSON file as an argument. For example:
```bash
python segmental.py test_files/radial_table/radial_table.json
```
This will load the radial_table.json file and create a segmental model based on the specified parameters.

Input JSON Format
The input JSON files are used to specify the parameters for the segmental model creation. The format is as follows:

```json
{
    "type": "3d",
    "source": "path/to/input/stl/file.stl",
    "export": "stl",
    "extrude_height": 12,
    "export_folder": "path/to/export/folder",
    "export_prefix": "prefix_for_exported_files",

    "params": {
        "type": "radial_slice" | "grid_slice",
        "transpose": 0,
        "height": 460,
        "thickness": 12,
        "kerf": 0.0,
        "length_slices": 3,
        "radial_slices" | "width_slices": 8,
        "vertical_offset": 0,
    }
}
```

The params section specifies the type of segmental model to create, as well as various parameters controlling the segmentation process.

### Dev Container
This project has a dev container specified in the .devcontainer directory. This is meant to be used with the Dev Containers VS Code extension. If the container is built, it will install all necessary requirements, as well as some VS Code extensions (only installed in the container) to make it easier to preview the stl files.

### Notes
The Segmental program is still a work in progress, and there may be bugs or incomplete features. Please report any issues or suggestions to the development team.

### Requirements
The program requires the following dependencies:

Python 3.x
NumPy
SciPy
matplotlib
Pillow
triangle

These dependencies can be installed using pip:

```bash
pip install -r requirements.txt
```