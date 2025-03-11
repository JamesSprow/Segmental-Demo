import cairo
from classes.helper_classes.shape import Shape, Offset, MultiShape

# Handles exporting shapes to svg files
class VectorExport:
    def __init__(self) -> None:
        pass

    # Exports shapes to individual svg files
    def export(self, shapes: list[Shape], file_name: str):
        counter = 1
        for shape in shapes:
            contour = shape.contour
            name = file_name + str(counter) + '.svg'
            self.export_piece(contour, name)
            counter += 1
    
    def transform_contour_to_origin(self, contour: tuple[tuple[float]]):
        puzzle_min_x = min([p[0] for p in contour])
        puzzle_min_y = min([p[1] for p in contour])

        contour = [[p[0] - puzzle_min_x, p[1] - puzzle_min_y] for p in contour]
        return contour
    

    def transform_contours_to_origin(self, contour: tuple[tuple[tuple[float]]]):
        puzzle_min_x = min([p[0] for p in contour[0]])
        puzzle_min_y = min([p[1] for p in contour[0]])
        for c in contour:
            if min([p[0] for p in c]) < puzzle_min_x:
                puzzle_min_x = max([p[0] for p in c])
            if max([p[1] for p in c]) < puzzle_min_y:
                puzzle_min_y = max([p[1] for p in c])
        
        return [[[p[0] - puzzle_min_x, p[1] - puzzle_min_y] for p in c] for c in contour]


    def draw_contour(self, contour, ctx: cairo.Context):
        #draw contour
        ctx.move_to(contour[0][0], contour[0][1])
        for i in contour:
            ctx.line_to(i[0], i[1])

        ctx.close_path()

    # Export individual contour to svg
    def export_piece(self, contour: tuple[tuple[float]], file_name: str):
        puzzle_width = None
        puzzle_total_height = None
        if len(contour) == 0:
            return
        if type(contour[0][0]) != float:
            contour = self.transform_contours_to_origin(contour)
            # get maximum bounds for contour
            puzzle_width = max([p[0] for p in contour[0]])
            puzzle_total_height = max([p[1] for p in contour[0]])
            for c in contour:
                if max([p[0] for p in c]) > puzzle_width:
                    puzzle_width = max([p[0] for p in c])
                if max([p[1] for p in c]) > puzzle_total_height:
                    puzzle_total_height = max([p[1] for p in c])
        else:
            contour = self.transform_contour_to_origin(contour)
            puzzle_width = max([p[0] for p in contour])
            puzzle_total_height = max([p[1] for p in contour])

        # creating a SVG surface, adding 1 pixel for clearance
        with cairo.SVGSurface(file_name,
                            puzzle_width + 1,
                            puzzle_total_height + 1
                            ) as surface: 
        
            # creating a cairo context object 
            ctx = cairo.Context(surface) 

            if type(contour[0][0]) != float:
                for c in contour:
                    self.draw_contour(c, ctx)
            else:
                self.draw_contour(contour, ctx)

            ctx.set_source_rgb(.75, 1, 0)
            #ctx.fill_preserve()
            ctx.set_line_width(0.04)

            ctx.stroke()

            surface.set_document_unit(6) #set units to mm
        
        # printing message when file is saved 
        print("File Saved:", file_name)
