# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:18:22 2018
Patterns generators
@author: oddvi
"""

import shapely.geometry
import shapely.affinity
import shapely.ops
"""
Visualizing
"""

def plotPolygon(polygon, interior_geom=True):
    import matplotlib.pyplot as plt
    fig = plt.figure(1, figsize=(5,5), dpi=90)
    ax = fig.add_subplot(111)
    ax.set_title('Polygon')
    ax.axis('equal')
    x,y = polygon.exterior.coords.xy
    ax.plot(x,y,'blue')
    if interior_geom == True:
        for interior in polygon.interiors:
            x,y = interior.coords.xy
            ax.plot(x,y,'blue')
            
"""
Generating regions
"""

def genTorsionFlexure(width_stem,length_flex,height_stem,width_flex):
    """
    Generation of the simple torsional flexure. Patterns can
    be generated through multiple transfomations...
                    __
     ______________|  | I stem length (d)
    |   ______________| I flex height (c)
    |__|<--------->|   length_flex (b)
    <--> width_stem (a)
    """
    a = width_stem; b = length_flex; c = height_stem; d = width_flex
    x1 = 0; y1 = c+d
    x2 = a+b; y2 = y1
    x3 = x2; y3 = 2*c+d
    x4 = 2*a+b; y4 = y3
    x5 = x4; y5 = c
    x6 = a; y6 = y5
    x7 = a; y7 = 0
    x8 = 0; y8 = 0
    x = [x1,x2,x3,x4,x5,x6,x7,x8]
    y = [y1,y2,y3,y4,y5,y6,y7,y8]
    tuple_coords = [(x[i],y[i]) for i in range(len(x))]
    return shapely.geometry.Polygon(tuple_coords)


def genYdX(solid_width,flexure_length,flexure_width,cut_width,thetaDeg):
    """
    Generation of YdX pattern. Pattern generated through pmm
    Returns a exterior ring of coorinates
    """
    import math
    a = solid_width; b = flexure_length; c = flexure_width; d = cut_width
    # Calculate repeating dimensions
    theta = math.radians(thetaDeg)
    bx=b*math.cos(theta)
    by=b*math.sin(theta)
    cxy=(c-c/math.cos(theta))/(-math.tan(theta))
    d2=d/2
    dxy=(d2-d/math.cos(theta))/(-math.tan(theta))
    dx=d*math.sin(theta)
    dy=d*math.cos(theta)
    w=a+bx+dxy+cxy+dx+a
    h=by+dy+c+d/2.0
    # Build array
    x0=0.0; y0=0.0
    x1=a+bx; y1=0.0
    x2=a; y2=by
    x3=a+dx; y3=by+dy
    x4=a+bx+dxy; y4=d2
    x5=w; y5=d2
    x6=w; y6=h
    x7=w-a-bx; y7=h
    x8=w-a; y8=h-by
    x9=w-a-dx; y9=h-by-dy
    x10=w-a-bx-dxy; y10=h-d2
    x11=0.0; y11=h-d2
    x12=0.0; y12=0.0
    x = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]
    y = [y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12]
    tuple_coords = [(x[i],y[i]) for i in range(len(x))]
    return shapely.geometry.Polygon(tuple_coords)

def genSquareCyclicSlit(cut_width,flexure_width,junction_length,edge_space,stem_width,num_flex,inside_start):
    """
    Generation of 1/8 of square cyclic slits. Full cell is generated with p4m.
    Returns a exterior ring of coorinates
    """
    import numpy as np
    a = cut_width; b = flexure_width; c = junction_length; d = edge_space; e = stem_width
    sqrt2 = 2**0.5
    ax = a/sqrt2/2 # x displacement along diagonal cut
    dx = a+b # displacement y direction
    dy = dx # displacement y direction
    h0 = c+a/2 # height in triangle
    l1 = a/2 # height baseline -> flexure bottom
    l2 = b+a/2 # height baseline -> flexure top
    x = np.array([])
    y = np.array([])
    
    if inside_start == True:
        x = np.append(x,h0) # 0
        y = np.append(y,h0) #
        x = np.append(x,l1) # 1
        y = np.append(y,l1) #
        x = np.append(x,h0-e) # 2
        y = np.append(y,l1) #
        x = np.append(x,h0-e) # 3
        y = np.append(y,0) #
        x = np.append(x,h0) # 4
        y = np.append(y,0) #
        x = np.append(x,h0) # 0
        y = np.append(y,h0) #
        insert_index = 4
                
    if inside_start == False: # outside start
        x = np.append(x,h0) # 0
        y = np.append(y,h0) #
        x = np.append(x,0) # 1
        y = np.append(y,0) #
        x = np.append(x,ax) # 2
        y = np.append(y,0) #
        x = np.append(x,ax+d) # 3
        y = np.append(y,0) #
        x = np.append(x,ax+d+l1) # 4
        y = np.append(y,l1) #
        x = np.append(x,h0) # 5
        y = np.append(y,l1) #
        x = np.append(x,h0) # 0
        y = np.append(y,h0) #
        insert_index = 3
                
    for n in range(num_flex):
        vec_x = np.array([])
        vec_y = np.array([])
        h = (n+1)*(a+b)+h0

        if inside_start==True: # build inside-out
            vec_x = np.append(vec_x, h-e) # 0
            vec_y = np.append(vec_y, l2)
            vec_x = np.append(vec_x, l2+ax) # 1
            vec_y = np.append(vec_y, l2)
            vec_x = np.append(vec_x, ax) # 2 
            vec_y = np.append(vec_y, 0)
            vec_x = np.append(vec_x, ax+d) # 3
            vec_y = np.append(vec_y, 0)
            vec_x = np.append(vec_x, ax+d+l1) # 4
            vec_y = np.append(vec_y, l1)
            vec_x = np.append(vec_x, h) # 5 
            vec_y = np.append(vec_y, l1)
            inside_start = False
        else: # build outside-in
            vec_x = np.append(vec_x, ax+l1) # 0
            vec_y = np.append(vec_y, l1)
            vec_x = np.append(vec_x, h-e) # 1
            vec_y = np.append(vec_y, l1)
            vec_x = np.append(vec_x, h-e) # 2
            vec_y = np.append(vec_y, 0)
            vec_x = np.append(vec_x, h) # 3
            vec_y = np.append(vec_y, 0)
            vec_x = np.append(vec_x, h) # 4
            vec_y = np.append(vec_y, l2)
            vec_x = np.append(vec_x, l2+ax+d) # 5
            vec_y = np.append(vec_y, l2)

            inside_start = True
        
        # shifts existing coordinates a distance dx and dy
        x += dx
        y += dy                
        # inserts new geometry from origo between the right coordinates
        x = np.concatenate((x[:insert_index],vec_x, x[insert_index:]),axis=0)
        y = np.concatenate((y[:insert_index],vec_y, y[insert_index:]),axis=0)
        # adds to index counter
        insert_index += 3
    tuple_coords = [(x[i],y[i]) for i in range(len(x))]
    return shapely.geometry.Polygon(tuple_coords)

    
    
"""
Swichback generator
"""

def genSwicback(num_turns, width_stem, length_flex, cut_width, width_flex):
    """" 
    NB! num_turns >= 1
    height = (cut_width+width_flex) * 2*num_turns
    width  = length_flex + 2*width_stem
    """
    height_stem = cut_width/2
    swichbacks = []
    dy = height_stem*2+width_flex
    # first segment
    start_segment = genTorsionFlexure(width_stem, length_flex, height_stem, width_flex)
    swichbacks.append(start_segment)
    # middle segment
    middle_segment = genTorsionFlexure(width_stem, length_flex - cut_width/2, height_stem, width_flex)
    # last segment
    middle_segment_mirror = shapely.affinity.scale(geom=middle_segment, xfact=-1, yfact=1, origin='center')
    end_segment = shapely.affinity.translate(start_segment, xoff=cut_width/2, yoff=dy*(num_turns*2))
    # adds number of desired turns
    for n in range(num_turns):
        if n == 0:
            swichbacks.append(shapely.affinity.translate(middle_segment_mirror, xoff=cut_width/2, yoff=dy*(n+1)))
        else:
            swichbacks.append(shapely.affinity.translate(middle_segment, xoff=cut_width/2, yoff=dy*(2*n)))
            swichbacks.append(shapely.affinity.translate(middle_segment_mirror, xoff=cut_width/2, yoff=dy*(2*n+1)))
    swichbacks.append(end_segment)
    unit = shapely.ops.cascaded_union(swichbacks)
    return unit


"""
Make unit cell of pattern
"""

def generatePmmUnit(generating_unit):
    [xmin, ymin, xmax, ymax] = generating_unit.bounds
    mirrored_x = shapely.affinity.scale(generating_unit, xfact=-1,yfact=1,origin=(xmax,ymin))
    mirrored_y = shapely.affinity.scale(generating_unit, xfact=1,yfact=-1,origin=(xmax,ymin))
    mirrored_xy = shapely.affinity.scale(generating_unit, xfact=-1,yfact=-1,origin=(xmax,ymin))
    unit_cell = shapely.ops.cascaded_union([generating_unit,mirrored_x,mirrored_y,mirrored_xy])
    unit_cell.buffer(0)
    return unit_cell

def generatePmUnit(generating_unit):
    [xmin, ymin, xmax, ymax] = generating_unit.bounds
    mirrored_y = shapely.affinity.scale(generating_unit, xfact=1,yfact=-1,origin=(ymin,xmin))
    unit_cell = shapely.ops.cascaded_union([generating_unit,mirrored_y])
    return unit_cell

def genP4mUnit(generating_unit):
    [xmin, ymin, xmax, ymax] = generating_unit.bounds
    mirrored_y = shapely.affinity.scale(generating_unit, xfact=-1,yfact=1,origin=(xmax,ymax))
    shapes = []
    shapes.append(generating_unit)
    shapes.append(mirrored_y)
    for i in range(1,4):
        shapes.append(shapely.affinity.rotate(generating_unit,angle=(90*i), origin=(xmax,ymax)))
        shapes.append(shapely.affinity.rotate(mirrored_y,angle=(90*i), origin=(xmax,ymax)))
    unit_cell = shapely.ops.cascaded_union(shapes)
    return unit_cell

    
"""
Master generator
Combines previous functions to create a unit of the desired pattern
"""

def makeInsideLET(width_stem,length_flex,height_stem,width_flex):
    flexure_gen = genTorsionFlexure(width_stem,length_flex,height_stem,width_flex)
    outside_let_unit = generatePmmUnit(flexure_gen)
    return outside_let_unit

def makeOutsideLET(width_stem,length_flex,height_stem,width_flex):
    import shapely.affinity
    flexure_gen = genTorsionFlexure(width_stem,length_flex,height_stem,width_flex)
    flexure_mirror_gen = shapely.affinity.scale(flexure_gen, xfact=-1, yfact=1, origin='center')
    outside_let_unit = generatePmmUnit(flexure_mirror_gen)
    return outside_let_unit

def makeYdX(solid_width,flexure_length,flexure_width,cut_width,thetaDeg):
    YdX_gen = genYdX(solid_width,flexure_length,flexure_width,cut_width,thetaDeg)
    YdX_unit = generatePmmUnit(YdX_gen)
    return YdX_unit

def makeSqCyclicSlitt(cut_width,flexure_width,junction_length,edge_space,stem_width,num_flex,inside_start):
    sq_cyclic_slit_gen = genSquareCyclicSlit(cut_width,flexure_width,junction_length,edge_space,stem_width,num_flex,inside_start)
    sq_cyclic_slit_unit = genP4mUnit(sq_cyclic_slit_gen)
    return sq_cyclic_slit_unit

def makeSqCyclicSlittFlexure(cut_width,flexure_width,junction_length,edge_space,stem_width,num_flex,inside_start):
    sq_cyclic_slit_gen = genSquareCyclicSlit(cut_width,flexure_width,junction_length,edge_space,stem_width,num_flex,inside_start)
    [xmin, ymin, xmax, ymax] = sq_cyclic_slit_gen.bounds
    sq_cyclic_slit_gen_mx = shapely.affinity.scale(geom=sq_cyclic_slit_gen, xfact=1.0, yfact=-1.0, zfact=1.0, origin=(xmax,ymin))
    sq_cyclic_slit_gen_my = shapely.affinity.scale(geom=sq_cyclic_slit_gen, xfact=-1.0, yfact=1.0, zfact=1.0, origin=(xmax,ymin))
    sq_cyclic_slit_gen_mxy = shapely.affinity.scale(geom=sq_cyclic_slit_gen, xfact=-1.0, yfact=-1.0, zfact=1.0, origin=(xmax,ymin))
    sq_cyclic_slit_flexure = shapely.ops.cascaded_union([sq_cyclic_slit_gen, sq_cyclic_slit_gen_mx, sq_cyclic_slit_gen_my, sq_cyclic_slit_gen_mxy])
    return sq_cyclic_slit_flexure

def makeSwitchbackPmm(num_turns, width_stem, length_flex, cut_width, width_flex):
    generating_unit = genSwicback(num_turns, width_stem, length_flex, cut_width, width_flex)
    generating_unit_rotated = shapely.affinity.rotate(generating_unit, angle=90, origin='center')
    unit = generatePmmUnit(generating_unit_rotated)
    return unit

"""
Generator and surface mapping
"""

def generatePmgLET(width_stem,length_flex,height_stem,width_flex, skew_angle, ncell_x, ncell_y):
    """
    Generates a pmg pattern with symmetry around y axis
    Main problem is the outline of the unit_cell after the skew deformation
    """
    # Generate generating_unit
    generating_unit = makeOutsideLET(width_stem,length_flex,height_stem,width_flex)
    # Skew and make pmg
    LET_skew = shapely.affinity.skew(generating_unit, xs=0, ys=skew_angle, origin=(0,0))
    LET_skew_mirror = shapely.affinity.scale(LET_skew, xfact=-1, yfact=1, origin=(0,0))
    unit_cell = shapely.ops.cascaded_union([LET_skew, LET_skew_mirror])
    unit_cell.buffer(0)
    # Maps the pattern
    bounds = generating_unit.bounds
    xmin = bounds[0]
    ymin = bounds[1]
    xmax = bounds[2]
    ymax = bounds[3]
    dx = (xmax-xmin)*2
    dy = ymax-ymin
    cell_duplicates = []
    for i in range(ncell_x):
        for j in range(ncell_y):
            cell_duplicates.append(shapely.affinity.translate(unit_cell, xoff=(dx*i), yoff=(dy*j)))
    surface_polygon = shapely.ops.cascaded_union(cell_duplicates)    
    return surface_polygon

def generateP2LET(width_stem,length_flex,height_stem,width_flex, skew_angle, ncell_x, ncell_y):
    """
    Generates a pmg pattern with symmetry around y axis
    Main problem is the outline of the unit_cell after the skew deformation
    """
    from math import tan
    from math import pi
    # Generate generating_unit
    generating_unit = makeOutsideLET(width_stem,length_flex,height_stem,width_flex)
    # Skew deformation
    unit_cell = shapely.affinity.skew(generating_unit, xs=skew_angle, ys=0, origin=(0,0))
    unit_cell.buffer(0)
    
    # Maps the pattern
    [xmin, ymin, xmax, ymax] = generating_unit.bounds
    dy = ymax-ymin
    dx = (xmax-xmin)
    dx_off = tan(skew_angle*(pi/180))*dy
    cell_duplicates = []
    for i in range(ncell_y):
        x_off=dx_off*i
        for j in range(ncell_x):
            cell_duplicates.append(shapely.affinity.translate(unit_cell, xoff=(x_off+dx*j), yoff=(dy*i)))
    surface_polygon = shapely.ops.cascaded_union(cell_duplicates)    
    return surface_polygon


"""
Surface mapping
"""

def mapSurface(polygon_cell, ncell_x, ncell_y, hex_cell=False):
    """
    Creates a nx x ny surface of units cells. For hexagonal units, hex_cell = True
    """
    [xmin, ymin, xmax, ymax] = polygon_cell.bounds
    dy = ymax-ymin
    dx = xmax-xmin
    if hex_cell == True:
        dx -= dy/(3**0.5)
    cell_duplicates = []
    for i in range(ncell_x):
        for j in range(ncell_y):
            cell_duplicates.append(shapely.affinity.translate(polygon_cell, xoff=(dx*i), yoff=(dy*j)))
    surface_polygon = shapely.ops.cascaded_union(cell_duplicates)    
    return surface_polygon

"""
Test calls for functions
"""

#Generators
#plotPolygon(genSwicback(num_turns=1, width_stem=1, length_flex=10, cut_width=1, width_flex=2))

#Units
plotPolygon(makeSqCyclicSlitt(cut_width=0.5, flexure_width=1 ,junction_length=3, edge_space=1.5, stem_width=1 ,num_flex=3 ,inside_start=False))
#plotPolygon(makeYdX(solid_width=1, flexure_length=5, flexure_width=1, cut_width = 0.5 ,thetaDeg=15))
#plotPolygon(makeOutsideLET(width_stem=1,length_flex=1,height_stem=1,width_flex=1))
#plotPolygon(makeInsideLET(width_stem=1,length_flex=1,height_stem=1,width_flex=1))
#plotPolygon(makeSwitchbackPmm(num_turns=1, width_stem=1, length_flex=10, cut_width=1, width_flex=2))

#Surfaces
#plotPolygon(generateP2LET(width_stem=1,length_flex=4,height_stem=0.2,width_flex=1,skew_angle=45,ncell_x=2, ncell_y=4))
#plotPolygon(11111111111111generatePmgLET(width_stem=1,length_flex=4,height_stem=0.2,width_flex=1,skew_angle=45,ncell_x=2, ncell_y=4))

#Flexures
#plotPolygon(makeSqCyclicSlittFlexure(cut_width=1, flexure_width=1 ,junction_length=1, edge_space=1.5, stem_width=1 ,num_flex=3 ,inside_start=True))






"""
Make polygon object from exterior and/or interior geometry
"""
def makePolygon(exterior, interior=False):
    """
    made redudant by:
    tuple_coords = [(x[i],y[i]) for i in range(len(x))]
    return shapely.geometry.Polygon(tuple_coords)
    """
    import shapely.geometry
    tuple_exterior_array = []
    for i in range(len(exterior[0])):
        tuple_exterior_array.append((exterior[0][i],exterior[1][i]))
    tuple_interior_array = []
    if interior != False:
        for interior_ring in interior:
            tuple_interior = []
            for j in range(len(interior_ring[0])):
                tuple_interior.append((interior_ring[0][j],interior_ring[1][j]))
            tuple_interior_array.append(tuple_interior)
    return shapely.geometry.Polygon(tuple_exterior_array, tuple_interior_array)


