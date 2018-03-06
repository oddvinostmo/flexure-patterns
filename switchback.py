# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:12:12 2018
Switchback square
@author: oddvi
"""
import matplotlib.pyplot as plt
import shapely.geometry
import shapely.affinity
import shapely.ops
import patternGenerators as gen

def make_square_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    """
    """
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
    Scales the square quart with 3**0.5 to match 30-120-30 triangle. addects edge_space and side_cut
    """
    xfact = 1.00001*3**0.5 # added tolerance for scaling
    edge_space /= xfact
    if side_cut == 'default':
        side_cut = cut_width/xfact / (2**0.5/2)
    sb_square_mod = make_square_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut)
    return shapely.affinity.scale(geom=sb_square_mod, xfact=xfact)

def make_hexagonal_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    """
    Scales the square quart with 1/3**0.5 to match 60-60-60 triangle. addects edge_space and side_cut
    """
    xfact = 1.00001/3**0.5 # added tolerance for scaling
    edge_space /= xfact
    if side_cut == 'default':
        side_cut = cut_width/xfact /(2**0.5/2)
    sb_square_mod = make_square_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default')
    return shapely.affinity.scale(geom=sb_square_mod, xfact=xfact)

"""
Tiles
"""

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




#gen.plotPolygon(make_triangular_switchback_gen_reg(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=4, side_cut=1))
#gen.plotPolygon(make_triangular_switchback_tile(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=2, side_cut=1))
#gen.plotPolygon(make_hexagonal_switchback_gen_reg(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=4, side_cut=1))
gen.plotPolygon(make_hexagonal_switchback_tile(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=5, side_cut='default'))
