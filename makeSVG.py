# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:52:16 2018

Run generations of patterns

@author: oddvi
"""

import patternGenerators as gen

"""
Save as SVG
"""

def make_SVG_string(polygon):
        """SVG representation for iPython notebook"""
        svg_top = '<svg xmlns="http://www.w3.org/2000/svg" ' \
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
        if polygon.is_empty:
            return svg_top + '/>'
        else:
            # Establish SVG canvas that will fit all the data + small space
            xmin, ymin, xmax, ymax = polygon.bounds
            if xmin == xmax and ymin == ymax:
                # This is a point; buffer using an arbitrary size
                xmin, ymin, xmax, ymax = polygon.buffer(1).bounds
            else:
                # Expand bounds by a fraction of the data ranges
                expand = 0.04  # or 4%, same as R plots
                widest_part = max([xmax - xmin, ymax - ymin])
                expand_amount = widest_part * expand
                xmin -= expand_amount
                ymin -= expand_amount
                xmax += expand_amount
                ymax += expand_amount
            dx = xmax - xmin
            dy = ymax - ymin
            width = dx
            height = dy
#            try:
#                scale_factor = max([dx, dy]) / max([width, height])
#            except ZeroDivisionError:
#                scale_factor = 1.
            scale_factor=1
            view_box = "{0} {1} {2} {3}".format(xmin, ymin, dx, dy)
            transform = "matrix(1,0,0,-1,0,{0})".format(ymax + ymin)
            return svg_top + (
                'width="{1}" height="{2}" viewBox="{0}" '
                'preserveAspectRatio="xMinYMin meet">'
                '<g transform="{3}">{4}</g></svg>'
                ).format(view_box, width, height, transform,
                         polygon.svg(scale_factor))

def make_SVG_file(polygon, name):
    polygon_svg = make_SVG_string(polygon)
    # polygon.svg(scale_factor=1.0)
    file=open(name,'w')
    polygon_svg = polygon_svg.replace('opacity="0.6"', 'opacity="1.0"')
    polygon_svg = polygon_svg.replace('fill="#66cc99"','fill="#b9b9b9"')
    polygon_svg = polygon_svg.replace('stroke-width="2.0"','stroke-width="0.1"')
    polygon_svg = polygon_svg.replace('stroke="#555555"','stroke="#000000"')
    file.write(polygon_svg)
    file.close()
    return



"""
Save flexures
"""
num = 3

# p6 patterns
import generateP6m_slitt_2_2 as genP6

# Unit
hex_unit_1 = genP6.makeHexCyclicSlitt(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=2,stem_width=1, num_flex=num, inside_start=False)
hex_unit_2 = genP6.makeHexCyclicSlitt_2(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=2,stem_width=1, num_flex=num , inside_start=False)
make_SVG_file(hex_unit_1,'hex_unit_1.svg')
make_SVG_file(hex_unit_2,'hex_unit_2.svg')

# Generating region
hex_gen_1 = genP6.genP6mSlit_2(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=2,stem_width=1,num_flex=num,inside_start=False)
make_SVG_file(hex_gen_1,'hex_gen_1.svg')

#Flexure
hex_flex_1 = genP6.makeHexCyclicSlittFlexure(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=2,stem_width=1,num_flex=num,inside_start=False)
hex_flex_2 = genP6.makeHexCyclicSlittFlexure_2(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=2,stem_width=1,num_flex=num,inside_start=False)
make_SVG_file(hex_flex_1,'hex_flex_1.svg')
make_SVG_file(hex_flex_2,'hex_flex_2.svg')


# p4m patterns
# Unit
sq_unit = gen.makeSqCyclicSlitt(cut_width=0.5, flexure_width=1 ,junction_length=3, edge_space=1.5, stem_width=1, num_flex=num, inside_start=False)
make_SVG_file(sq_unit,'sq_unit.svg')

# Flexure
sq_flex = gen.makeSqCyclicSlittFlexure(cut_width=0.5, flexure_width=1 ,junction_length=3, edge_space=1.5, stem_width=1, num_flex=num, inside_start=False)
make_SVG_file(sq_flex,'sq_flex.svg')



# cmm patterns
YdX_unit = gen.makeYdX(solid_width=1, flexure_length=5, flexure_width=1, cut_width = 0.5 ,thetaDeg=15)
make_SVG_file(YdX_unit,'YdX_unit.svg')

# pmm patterns
swbk_unit = gen.genSwicback(num_turns=1, width_stem=1, length_flex=10, cut_width=1, width_flex=2)
make_SVG_file(swbk_unit,'swbk_unit.svg')


# Flexures
outside_LET = gen.makeOutsideLET(width_stem=1,length_flex=10,height_stem=0.5,width_flex=1)
make_SVG_file(outside_LET,'outside_LET.svg')

inside_LET = gen.makeInsideLET(width_stem=1,length_flex=10,height_stem=0.5, width_flex=1)
make_SVG_file(inside_LET,'inside_LET.svg')

swtchbk = gen.makeSwitchbackPmm(num_turns=1, width_stem=1, length_flex=10, cut_width=1, width_flex=2)
make_SVG_file(swtchbk,'swtchbk.svg')


# Pattern examples
sq_pat = gen.mapSurface(polygon_cell=sq_unit, ncell_x=3, ncell_y=2, hex_cell=False)
make_SVG_file(sq_pat,'sq_pat.svg')

hex_pat_1 = gen.mapSurface(polygon_cell=hex_unit_1, ncell_x=3, ncell_y=2, hex_cell=True)
make_SVG_file(hex_pat_1,'hex_pat_1.svg')

#hex_pat_2 = gen.mapSurface(polygon_cell=hex_unit_2, ncell_x=3, ncell_y=2, hex_cell=True)
#make_SVG_file(hex_pat_2,'hex_pat_2.svg')

