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
    """Plots a shapely polygon using the matplotlib library """
    import matplotlib.pyplot as plt
    if polygon.type == 'MultiPolygon': 
        print(polygon.type)
        polygon = polygon.buffer(0)
        if polygon.type=='MultPolygon': polygon = polygon.buffer(0.0001, resolution=1) 
    try:
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
    except:
        print('Plotting failed of {0}'.format(polygon.type))
        
def get_l1_l2(polygon):
    xmin, ymin, xmax, ymax = polygon.bounds
    return xmax-xmin, ymax-ymin
            
"""
Generating regions
"""

def make_torsion_flexure(width_stem,length_flex,height_stem,width_flex):
    """
    Generation of the simple torsional flexure. Patterns can
    be generated through multiple transfomations...
                    __
     ______________|  | I stem length (d)
    |   ______________| I flex height (c)
    |__|<--------->|   length_flex (b)
    <--> width_stem (a)
    l1, l2, angle = 2*a+b, 2*c+d, 90
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
    coords = [(x[i],y[i]) for i in range(len(x))]
    # alternative: coords = list(zip(a,b))
    return shapely.geometry.Polygon(coords)


def make_ydx_gen_reg(solid_width,flexure_length,flexure_width,cut_width,thetaDeg):
    """
    Full unit generated through pmm
    l1, l2, angle = w, h, 90
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
    coords = [(x[i],y[i]) for i in range(len(x))]
    return shapely.geometry.Polygon(coords)


"""
LET generators
"""

def make_square_let_gen_reg(cut_width, flexure_width, junction_length, 
                            edge_space, stem_width, num_flex, inside_start):
    """
    Generation of 1/8 of square cyclic slits. Full cell is generated with p4m.
    Returns a exterior ring of coorinates
    l1, l2, angle = h, h, 45
    """
    import numpy as np
    a = cut_width; b = flexure_width; c = junction_length; d = edge_space; e = stem_width
    sqrt2 = 2**0.5
    ax = a/sqrt2/2.0 # x displacement along diagonal cut
    dx = a+b # displacement y direction
    dy = dx # displacement y direction
    h0 = c+a/2.0 # height in triangle
    l1 = a/2.0 # height baseline -> flexure bottom
    l2 = b+a/2.0 # height baseline -> flexure top
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
        x = np.append(x,0+ax/2) # 1
        y = np.append(y,0+ax/2) # 1
        x = np.append(x,0+ax) # 1'
        y = np.append(y,0) # 1'
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
        insert_index = 4
                
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
    coords = [(x[i],y[i]) for i in range(len(x))]
    return shapely.geometry.Polygon(coords)


"""
Triangular versons are the 120-deg version that create triangular tiles
"""

def make_triangular_let_gen_reg(cut_width, flexure_width, junction_length, 
               edge_space, stem_width, num_flex, inside_start): # was not the 
    """
    Generates 1/12 of a LET that maps with the p6m group
    """
    import numpy as np
    a = cut_width; b = flexure_width; c = junction_length; d = edge_space; e = stem_width
    sqrt3 = 3**0.5
    dy = a+b # displacement y direction
    dx = sqrt3*dy # displacement x direction
    h0 = c+a/2.0 # height in triangle
    ax = a*sqrt3/2 # x displacement along diagonal cut
    l1 = a/2.0 # height baseline -> flexure bottom
    l2 = b+a/2.0 # height baseline -> flexure top
    tol = 0.0001
    x = np.array([])
    y = np.array([])
    
    if inside_start == True: # inside start
        x = np.append(x,[sqrt3*h0]) # x0
        y = np.append(y,[h0]) #y0
        x = np.append(x,[sqrt3*a/2-tol]) # x1
        y = np.append(y,[a/2]) #y1
        x = np.append(x,[sqrt3*h0-e]) #x2
        y = np.append(y,[a/2]) #y2
        x = np.append(x,[sqrt3*h0-e]) #x3
        y = np.append(y,[0]) #y3
        x = np.append(x,[sqrt3*h0]) #x4
        y = np.append(y,[0]) # y4
        x = np.append(x,[sqrt3*h0]) # x0
        y = np.append(y,[h0]) #y0
        insert_index = 4
                
    if inside_start == False: # outside start
        x = np.append(x,[sqrt3*h0]) # x0
        y = np.append(y,[h0]) # y0
        x = np.append(x,[3*ax/4-tol]) # x1
        y = np.append(y,[sqrt3*ax/4]) #y1
        x = np.append(x,[ax]) # x1'
        y = np.append(y,[0]) #y1'
        x = np.append(x,[d+ax]) # x2
        y = np.append(y,[0]) # y2
        x = np.append(x,[d+ax]) #x3 
        y = np.append(y,[a/2]) # y3
        x = np.append(x,[sqrt3*h0]) # x4
        y = np.append(y,[a/2]) # y4
        x = np.append(x,[sqrt3*h0]) # x0
        y = np.append(y,[h0]) # y0
        insert_index = 3
                
    for n in range(num_flex):
        vec_x = np.array([])
        vec_y = np.array([])
        h = (n+1)*(a+b)+c+a/2
        
        if inside_start==True: # build inside-out
            vec_x = np.append(vec_x, [sqrt3*h-e]) # x1
            vec_y = np.append(vec_y, [l2]) # 1
            vec_x = np.append(vec_x, [sqrt3*l2+ax]) # 2
            vec_y = np.append(vec_y, [l2]) # 2
            vec_x = np.append(vec_x, [ax]) # 3
            vec_y = np.append(vec_y, [0]) # 3
            vec_x = np.append(vec_x, [d+ax]) # 4, can remove 2* for sharp cuts !! affects corner
            vec_y = np.append(vec_y, [0]) # 4
            vec_x = np.append(vec_x, [ax+d]) # 5, add +l1*sqrt3 for drafted corner
            vec_y = np.append(vec_y, [l1]) # 5
            vec_x = np.append(vec_x, [sqrt3*h]) # 6
            vec_y = np.append(vec_y, [l1]) # 6
            inside_start = False
        else: # build outside-in
            vec_x = np.append(vec_x, [ax+sqrt3*l1])   # x = sqrt3*a/2 for N = 2 and ins=False, #should be 2* for ins = True
            vec_y = np.append(vec_y, [l1])
            vec_x = np.append(vec_x, [sqrt3*h-e])
            vec_y = np.append(vec_y, [l1])
            vec_x = np.append(vec_x, [sqrt3*h-e])
            vec_y = np.append(vec_y, [0])
            vec_x = np.append(vec_x, [sqrt3*h])
            vec_y = np.append(vec_y, [0])
            vec_x = np.append(vec_x, [sqrt3*h])
            vec_y = np.append(vec_y, [l2])
            vec_x = np.append(vec_x, [sqrt3*(b+a)+d+ax]) #6 can remove 2* for sharp cuts
            vec_y = np.append(vec_y, [l2])
            inside_start = True
        
        # shifts existing coordinates a distance dx and dy
        x += dx
        y += dy                
        # inserts new geometry from origo between the right coordinates
        x = np.concatenate((x[:insert_index],vec_x, x[insert_index:]),axis=0)
        y = np.concatenate((y[:insert_index],vec_y, y[insert_index:]),axis=0)
        # adds to index counter
        insert_index += 3
    coords = [(x[i],y[i]) for i in range(len(x))]
    return shapely.geometry.Polygon(coords)


def make_p6m_unit(generating_region):
    """
    The transformations done on a generating unit to make the p6m pattern
    Notes: works only for 120 deg flexure
    """
    xmin, ymin, xmax, ymax = generating_region.bounds
    mirrored_y = shapely.affinity.scale(generating_region, xfact=-1,yfact=1,
                                        origin=(xmax,ymax))
    shapes = []
    shapes.append(generating_region)
    shapes.append(mirrored_y)
    for i in range(1,4):
        shapes.append(shapely.affinity.rotate(generating_region,angle=(120*i),
                                              origin=(xmax,ymax)))
        shapes.append(shapely.affinity.rotate(mirrored_y,angle=(120*i), 
                                              origin=(xmax,ymax)))
    unit_cell_half = shapely.ops.cascaded_union(shapes)
    # duplicate unit_cell_half
    unit_cell_half_rotated = shapely.affinity.rotate(unit_cell_half,angle=(300), 
                                                     origin=(xmax*2,ymin))
    unit_cell = shapely.ops.cascaded_union([unit_cell_half, unit_cell_half_rotated])
    return unit_cell


def make_triangular_let_flexure(cut_width, flexure_width, 
            junction_length, edge_space, stem_width, num_flex, inside_start):
    """
    Notes: works only for 30-120-30 deg generating region
    """
    generating_region = make_triangular_let_gen_reg(cut_width, flexure_width, 
            junction_length, edge_space, stem_width, num_flex, inside_start)
    xmin, ymin, xmax, ymax = generating_region.bounds
    mirrored_y = shapely.affinity.scale(generating_region, xfact=-1, yfact=1, origin=(xmax,ymin))
    mirrored_x = shapely.affinity.scale(generating_region, xfact=1, yfact=-1, origin=(xmax,ymin))
    mirrored_xy =shapely.affinity.scale(generating_region, xfact=-1, yfact=-1, origin=(xmax,ymin))
    shapes = []
    shapes.append(generating_region)
    shapes.append(mirrored_y)
    shapes.append(mirrored_x)
    shapes.append(mirrored_xy)
    unit_cell_flex = shapely.ops.cascaded_union(shapes)
    return unit_cell_flex


def make_triangular_let_unit(cut_width, flexure_width, junction_length, edge_space, stem_width, num_flex, inside_start):
    """
    Notes: works only for 30-120-30 deg generating region
    l1, l2, angle = 2*a+b, 2*c+d, 90
    """
    generating_region = make_triangular_let_gen_reg(cut_width, flexure_width,junction_length, 
                                       edge_space,stem_width, num_flex, inside_start)
    triangular_let_unit = make_p6m_unit(generating_region)
#    xmin, ymin, xmax, ymax = triangular_let_unit.bounds
#    height = ymax-ymin # junction_length + num_flex*(flexure_width+cut_width)
#    l1 = 2*height / (3**0.5)
#    l2 = l1
#    angle = 60
#    lattice_dim = {'l1':l1, 'l2':l2, 'angle':angle}
    return triangular_let_unit.buffer(0)


"""
Hexagonal versons are the 60-deg version
These are scaled and not allways 100 %
"""


def make_hexagonal_let_gen_reg(cut_width, flexure_width, junction_length, edge_space, stem_width, num_flex, inside_start):
    """
    Generation of 1/8 of hexagonal cyclic slits. Full cell is generated with p6m.
    Returns a exterior ring of coorinates
    s = scaled, r = rotated, m = mirrored, t = translated
    dimensions are scaled - fix!!
    """
    edge_space *= 3
    stem_width *= 3
    generating_unit = make_triangular_let_gen_reg(cut_width, flexure_width,junction_length, edge_space, stem_width, num_flex, inside_start) # have to add a edge cut parameter...
    p6m_sm = shapely.affinity.scale(geom=generating_unit, xfact=(-1/3), yfact=1, zfact=1, origin='center')
    p6m_smr = shapely.affinity.rotate(geom=p6m_sm, angle=90, origin='center')
    xmin, ymin, xmax, ymax = p6m_smr.bounds  
    p6m_smrt = shapely.affinity.translate(geom=p6m_smr, xoff = -xmin, yoff=-ymin)
    return p6m_smrt


def make_hexagonal_let_flexure(cut_width, flexure_width, junction_length, 
                              edge_space, stem_width, num_flex, inside_start):
    """
    Makes the flexure that is maped to the hexagonal lattice
    Notes:  - works only for 60 deg flexure
            - uses the regualr ... might have some issues with some lengths
    """
    edge_space *= 3
    stem_width *= 3
    triangular_let_flexure = make_triangular_let_flexure(cut_width, flexure_width, 
                                                     junction_length, edge_space, 
                                                     stem_width, num_flex, 
                                                     inside_start)
    hex_cyclic_slit_flexure = shapely.affinity.scale(geom=triangular_let_flexure, 
                                                       xfact=(1/3), yfact=1, zfact=1, 
                                                       origin='center')
    return hex_cyclic_slit_flexure


def make_hexagonal_let_unit(cut_width, flexure_width, junction_length, 
                       edge_space, stem_width, num_flex, inside_start):
    """
    Real input length is not accurate, multiply something with 3 to make accurate measures...
    """
    edge_space *= 3
    stem_width *= 3
    generating_unit = make_hexagonal_let_gen_reg(cut_width, flexure_width, junction_length, edge_space, stem_width, num_flex, inside_start)
    xmin, ymin, xmax, ymax = generating_unit.bounds 
    gen_mx = shapely.affinity.scale(geom=generating_unit, xfact=1.0, yfact=-1.0, zfact=1.0, origin=(xmax,ymin))
    gen_my = shapely.affinity.scale(geom=generating_unit, xfact=-1.0, yfact=1.0, zfact=1.0, origin=(xmax,ymin))
    gen_mxy = shapely.affinity.scale(geom=generating_unit, xfact=-1.0, yfact=-1.0, zfact=1.0, origin=(xmax,ymin))
    mirror = shapely.ops.cascaded_union([gen_mx, gen_mxy])
    xmin, ymin, xmax, ymax = mirror.bounds
    mirror_rot1 = shapely.affinity.rotate(geom=mirror, angle=60, origin=(xmin,ymax))
    mirror_rot2 = shapely.affinity.rotate(geom=mirror, angle=-60, origin=(xmax,ymax))
    one_half = shapely.ops.cascaded_union([mirror_rot1, mirror_rot2, generating_unit, gen_my])
    xmin, ymin, xmax, ymax = one_half.bounds
    other_half = shapely.affinity.rotate(geom=one_half, angle=-60, origin=(xmax,ymin))
    unit = shapely.ops.cascaded_union([one_half, other_half])
    return unit
    
"""
Swichback generator
"""

def make_rectangular_switchback_gen_reg(num_turns, width_stem, length_flex, cut_width, width_flex):
    """" 
    NB! num_turns >= 1
    height = (cut_width+width_flex) * 2*num_turns
    width  = length_flex + 2*width_stem
    """
    height_stem = cut_width/2
    swichbacks = []
    dy = height_stem*2+width_flex
    # first segment
    start_segment = make_torsion_flexure(width_stem, length_flex, height_stem, width_flex)
    swichbacks.append(start_segment)
    # middle segment
    middle_segment = make_torsion_flexure(width_stem, length_flex - cut_width/2, height_stem, width_flex)
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
    return shapely.ops.cascaded_union(swichbacks)

def make_square_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    """ Tile and unit is equal and made through 4 rotations of the top of the triangle """
    import numpy as np
    a = cut_width; b = flexure_width; c = junction_length; d = edge_space
    if side_cut == 'default': # x displacement along diagonal cut
        ax = cut_width/(2**0.5)/2    
    else:
        ax = side_cut
    dx = a+b # displacement y direction
    dy = dx # displacement y direction
    h0 = a+b/2+c # height in triangle
    l1 = b/2 # height baseline -> flexure bottom
    l2 = a+b/2 # height baseline -> flexure top
    x = np.array([])
    y = np.array([])
    x = np.append(x, 0) # 0
    y = np.append(y, h0) # 0
    x = np.append(x, -h0+l2+ax/2) # 1
    y = np.append(y, l2+ax/2) # 1
    x = np.append(x, -h0+l2+ax) # 2
    y = np.append(y, l2) # 2
    x = np.append(x, -h0+ax) # 3
    y = np.append(y, 0) # 3
    x = np.append(x, h0-ax) # 4
    y = np.append(y, 0) # 4
    x = np.append(x, h0-l1-ax) # 5
    y = np.append(y, l1) # 5
    x = np.append(x, -h0+l1+d+ax) # 6
    y = np.append(y, l1) # 6
    x = np.append(x, -h0+l2+d+ax) # 7
    y = np.append(y, l2) # 7
    x = np.append(x, h0-l2-ax) # 8
    y = np.append(y, l2) # 8
    x = np.append(x, h0-l2-ax/2) # 9
    y = np.append(y, l2+ax/2) # 9
    x = np.append(x, 0) # 0
    y = np.append(y, h0) # 0
    insert_index = 4
    for n in range(num_flex):
        h = (n+1)*(a+b)+h0
        vec_x = np.array([])
        vec_y = np.array([])
        vec_x = np.append(vec_x, -h+l2+ax) # 0
        vec_y = np.append(vec_y, l2)
        vec_x = np.append(vec_x, h-l2-ax-d) # 1
        vec_y = np.append(vec_y, l2)
        vec_x = np.append(vec_x, h-l1-d-ax) # 2 
        vec_y = np.append(vec_y, l1)
        vec_x = np.append(vec_x, -h+l1+ax) # 3
        vec_y = np.append(vec_y, l1)
        vec_x = np.append(vec_x, -h+ax) # 4
        vec_y = np.append(vec_y, 0)
        vec_x = np.append(vec_x, h-ax) # 5 
        vec_y = np.append(vec_y, 0)
        if n%2:
            vec_x = -vec_x
            vec_x = np.flipud(vec_x)
            vec_y = np.flipud(vec_y)
            insert_index += 4
        y += dy # shifts existing coordinates a distance dy
        x = np.concatenate((x[:insert_index],vec_x, x[insert_index:]),axis=0) # inserts new geometry from origo between the right coordinates
        y = np.concatenate((y[:insert_index],vec_y, y[insert_index:]),axis=0)    
        insert_index +=1 # adds to index counter
    coords = [(x[i],y[i]) for i in range(len(x))]
    return shapely.geometry.Polygon(coords)

def make_triangular_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    """ 
    Scales the triangle with 3**0.5 to match 30-120-30 triangle. 
    For correct values, edge_space and side_cut are also scaled
    """
    xfact = 1.00001*3**0.5 # added tolerance for scaling
    edge_space /= xfact
    if side_cut == 'default':
        side_cut = cut_width/xfact / (2**0.5/2)
    sb_square_mod = make_square_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut)
    return shapely.affinity.scale(geom=sb_square_mod, xfact=xfact)

def make_hexagonal_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    """
    Scales the square quart with 1/3**0.5 to match 60-60-60 triangle. 
    addects edge_space and side_cut
    """
    xfact = 1.00001/3**0.5 # added tolerance for scaling
    edge_space /= xfact
    if side_cut == 'default':
        side_cut = cut_width/xfact /(2**0.5/2)
    sb_square_mod = make_square_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default')
    return shapely.affinity.scale(geom=sb_square_mod, xfact=xfact)


"""
Make unit cell of pattern
"""

def make_pmm_unit(generating_unit):
    [xmin, ymin, xmax, ymax] = generating_unit.bounds
    mirrored_x = shapely.affinity.scale(generating_unit, xfact=-1,yfact=1,origin=(xmax,ymin))
    mirrored_y = shapely.affinity.scale(generating_unit, xfact=1,yfact=-1,origin=(xmax,ymin))
    mirrored_xy = shapely.affinity.scale(generating_unit, xfact=-1,yfact=-1,origin=(xmax,ymin))
    unit_cell = shapely.ops.cascaded_union([generating_unit,mirrored_x,mirrored_y,mirrored_xy])
    unit_cell.buffer(0)
    return unit_cell

def make_pm_unit(generating_unit):
    [xmin, ymin, xmax, ymax] = generating_unit.bounds
    mirrored_y = shapely.affinity.scale(generating_unit, xfact=1,yfact=-1,origin=(ymin,xmin))
    unit_cell = shapely.ops.cascaded_union([generating_unit,mirrored_y])
    return unit_cell

def make_p4m_unit(generating_unit):
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

# Flexures

def make_inside_let(width_stem,length_flex,height_stem,width_flex):
    flexure_gen = make_torsion_flexure(width_stem,length_flex,height_stem,width_flex)
    inside_let = make_pmm_unit(flexure_gen)
    return inside_let

def make_outside_let(width_stem,length_flex,height_stem,width_flex):
    import shapely.affinity
    flexure_gen = make_torsion_flexure(width_stem,length_flex,height_stem,width_flex)
    flexure_mirror_gen = shapely.affinity.scale(flexure_gen, xfact=-1, yfact=1, origin='center')
    outside_let = make_pmm_unit(flexure_mirror_gen)
    return outside_let

# Units

def make_ydx_unit(solid_width,flexure_length,flexure_width,cut_width,thetaDeg):
    YdX_gen = make_ydx_gen_reg(solid_width,flexure_length,flexure_width,cut_width,thetaDeg)
    YdX_unit = make_pmm_unit(YdX_gen)
    return YdX_unit

def make_square_let_unit(cut_width,flexure_width,junction_length,edge_space,stem_width,num_flex,inside_start):
    sq_cyclic_slit_gen = make_square_let_gen_reg(cut_width,flexure_width,junction_length,edge_space,stem_width,num_flex,inside_start)
    sq_cyclic_slit_unit = make_p4m_unit(sq_cyclic_slit_gen)
    return sq_cyclic_slit_unit

def make_rectangular_switchback_unit(num_turns, width_stem, length_flex, cut_width, width_flex):
    generating_region = make_rectangular_switchback_gen_reg(num_turns, width_stem, length_flex, cut_width, width_flex)
    return make_pmm_unit(generating_region)

# Tiles

def make_triangular_switchback_tile(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    sb_third = make_triangular_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut)
    xmin, ymin, xmax, ycoord = sb_third.bounds
    xcoord = (xmax+xmin)/2
    sb_third_r120 = shapely.affinity.rotate(geom=sb_third, angle=120, origin=(xcoord,ycoord))
    sb_third_r240 = shapely.affinity.rotate(geom=sb_third, angle=240, origin=(xcoord,ycoord))
    return shapely.ops.cascaded_union([sb_third, sb_third_r120, sb_third_r240])

def make_square_switchback_tile(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    sb_quart = make_square_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut)
    xmin, ymin, xmax, ycoord = sb_quart.bounds # ycoord is ymax
    xcoord = (xmax+xmin)/2
    sb_quart_r90 = shapely.affinity.rotate(geom=sb_quart, angle=90, origin=(xcoord,ycoord))
    sb_quart_r180 = shapely.affinity.rotate(geom=sb_quart, angle=180, origin=(xcoord,ycoord))
    sb_quart_r270 = shapely.affinity.rotate(geom=sb_quart, angle=270, origin=(xcoord,ycoord))
    return shapely.ops.cascaded_union([sb_quart, sb_quart_r90, sb_quart_r180, sb_quart_r270])

def make_hexagonal_switchback_tile(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    sb_sixt = make_hexagonal_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut)
    xmin, ymin, xmax, ycoord = sb_sixt.bounds
    xcoord = (xmax+xmin)/2
    sb_rotated = [sb_sixt]
    for n in range(6):
        sb_rotated.append(shapely.affinity.rotate(geom=sb_sixt, angle=60*(1*n), origin=(xcoord,ycoord)))
    return shapely.ops.cascaded_union(sb_rotated)


"""
Generator and surface mapping
"""

def map_pmg_let(width_stem,length_flex,height_stem,width_flex, skew_angle, ncell_x, ncell_y):
    """
    Generates a pmg pattern with symmetry around y axis
    Main problem is the outline of the unit_cell after the skew deformation
    """
    # Generate generating_unit
    generating_unit = make_outside_let(width_stem,length_flex,height_stem,width_flex)
    # Skew and make pmg
    LET_skew = shapely.affinity.skew(generating_unit, xs=0, ys=skew_angle, origin=(0,0))
    LET_skew_mirror = shapely.affinity.scale(LET_skew, xfact=-1, yfact=1, origin=(0,0))
    unit_cell = shapely.ops.cascaded_union([LET_skew, LET_skew_mirror])
    unit_cell.buffer(0)
    # Maps the pattern
    [xmin, ymin, xmax, ymax] = generating_unit.bounds
    dx = (xmax-xmin)*2
    dy = ymax-ymin
    cell_duplicates = []
    for i in range(ncell_x):
        for j in range(ncell_y):
            cell_duplicates.append(shapely.affinity.translate(unit_cell, xoff=(dx*i), yoff=(dy*j)))
    surface_polygon = shapely.ops.cascaded_union(cell_duplicates)    
    return surface_polygon

def map_p2_let(width_stem, length_flex, height_stem, width_flex, skew_angle, ncell_x, ncell_y):
    """
    Generates a pmg pattern with symmetry around y axis
    Main problem is the outline of the unit_cell after the skew deformation
    """
    from math import tan, pi
    # Generate generating_unit
    generating_unit = make_outside_let(width_stem,length_flex,height_stem,width_flex)
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

def map_surface(unit_cell, ncell_x, ncell_y, lattice='rectangular'):
    """
    Creates a nx x ny surface of units cells. For hexagonal units, hex_cell = True
    """
    [xmin, ymin, xmax, ymax] = unit_cell.bounds
    dy = ymax-ymin #-0.00001 # there are some nummerical noise, so this is a temp bug fix...
    dx = xmax-xmin
    cell_duplicates = []
    if lattice == 'rectangular':
        for i in range(ncell_y):
            for j in range(ncell_x):
                cell_duplicates.append(shapely.affinity.translate(unit_cell, xoff=(dx*j), yoff=(dy*i)))
                
    elif lattice == 'rhombic':
        dx = 2*dy/(3**0.5)
        for i in range(ncell_y):
            DX=i*dx/2 # is the addition for each row up
            for j in range(ncell_x):
                cell_duplicates.append(shapely.affinity.translate(unit_cell, xoff=(dx*j+DX), yoff=(dy*i)))
                
    surface_polygon = combine_borders(cell_duplicates)    
    return surface_polygon

def combine_borders(geoms):
    return shapely.ops.cascaded_union([
            geom.buffer(0.00001, resolution=1) if geom.type =='MultiPolygon' 
            else geom for geom in geoms])

    
class GenReg():
    
    def __init__(self):
        
    

"""
Test calls for functions can be done by uncommenting the following lines
"""

#Generators
#plotPolygon(make_rectangular_switchback_gen_reg(num_turns=1, width_stem=1, length_flex=10, cut_width=1, width_flex=2))
#
##Units
#plotPolygon(make_ydx_unit(solid_width=1, flexure_length=5, flexure_width=1, cut_width = 0.5 ,thetaDeg=45))

## LET
#plotPolygon(make_outside_let(width_stem=1,length_flex=1,height_stem=1,width_flex=1))
#plotPolygon(make_inside_let(width_stem=1,length_flex=1,height_stem=1,width_flex=1))

# Square LET
#plotPolygon(make_square_let_gen_reg(cut_width=1, flexure_width=1 ,junction_length=3, edge_space=1.5, stem_width=1, num_flex=3, inside_start=False))
#plotPolygon(make_square_let_unit(cut_width=1, flexure_width=1, junction_length=3, edge_space=1.5, stem_width=1, num_flex=3, inside_start=False))

# Triangular LET
#plotPolygon(make_triangular_let_gen_reg(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=2,stem_width=1,num_flex=3,inside_start=False))
#plotPolygon(make_triangular_let_unit(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=2,stem_width=1,num_flex=3,inside_start=False))
#plotPolygon(make_triangular_let_flexure(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=1,stem_width=1,num_flex=3,inside_start=False))

# Hexagonal LET
#plotPolygon(make_hexagonal_let_gen_reg(cut_width=1, flexure_width=2, junction_length=6, edge_space=4, stem_width=2, num_flex=3, inside_start=False)) # 
#plotPolygon(make_hexagonal_let_unit(cut_width=1, flexure_width=2, junction_length=6, edge_space=1, stem_width=1, num_flex=3, inside_start=False)) # BUG fail due to numerical distortions
#plotPolygon(make_hexagonal_let_flexure(cut_width=0.5, flexure_width=1, junction_length=3, edge_space=1, stem_width=1, num_flex=3, inside_start=False)) # not reliable (due to scaling?)


## Switchbacks
## Comment: a little tweaking is necessary on the 'default'
#plotPolygon(make_triangular_switchback_gen_reg(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=4, side_cut=1))
#plotPolygon(make_hexagonal_switchback_gen_reg(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=4, side_cut=1))
#plotPolygon(make_hexagonal_switchback_tile(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=5, side_cut='default'))
#plotPolygon(make_rectangular_switchback_unit(num_turns=1, width_stem=1, length_flex=10, cut_width=1, width_flex=2))
#plotPolygon(make_triangular_switchback_tile(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=2, side_cut=1))
#plotPolygon(make_square_switchback_tile(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=2, side_cut=1))
#plotPolygon(make_hexagonal_switchback_tile(cut_width=0.5, flexure_width=1, junction_length=2, edge_space=1, num_flex=3, side_cut=1))

#Surfaces
#plotPolygon(map_p2_let(width_stem=1,length_flex=4,height_stem=0.2,width_flex=1,skew_angle=45,ncell_x=2, ncell_y=4))
#plotPolygon(map_pmg_let(width_stem=1, length_flex=4, height_stem=0.2, width_flex=1, skew_angle=45, ncell_x=2, ncell_y=10))
#
#hex_unit = make_triangular_let_unit(cut_width=1,flexure_width=2,junction_length=4,edge_space=4,stem_width=1,num_flex=3,inside_start=False)
#mapped_hex = map_surface(unit_cell=hex_unit, ncell_x=3, ncell_y=3, lattice='rhombic')
#plotPolygon(mapped_hex)


#TEST CALL
#polygon = make_square_let_unit(cut_width=1, flexure_width=1, junction_length=3, edge_space=1.5, stem_width=1, num_flex=3, inside_start=False)












