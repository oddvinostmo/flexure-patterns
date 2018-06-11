# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:52:16 2018

Run generations of patterns

@author: oddvi
"""

#import patternGenerators as gen

import shapely.geometry
import shapely.affinity
import shapely.ops

"""
Save as SVG
"""

def makeSVGstring(polygon):
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
            
def scaleToPage(polygon, pref):
    xmin, ymin, xmax, ymax = polygon.bounds
    dx = xmax-xmin
    return shapely.affinity.scale(polygon, xfact=pref/dx , yfact=pref/dx)

def createSVG(polygon, name, width):
    polygon_scaled = scaleToPage(polygon, width)
    polygon_svg = makeSVGstring(polygon_scaled)
    # polygon.svg(scale_factor=1.0)
    file=open(name,'w')
    polygon_svg = polygon_svg.replace('opacity="0.6"', 'opacity="1.0"')
#    polygon_svg = polygon_svg.replace('fill="#66cc99"','fill="#b9b9b9"')
    polygon_svg = polygon_svg.replace('stroke-width="2.0"','stroke-width="0.265"')
#    polygon_svg = polygon_svg.replace('stroke="#555555"','stroke="#000000"')
    file.write(polygon_svg)
    file.close()
    return


    


"""
Save
"""

if __name__ == '__main__':
    'smthng'
#    p = scaleToPage()x