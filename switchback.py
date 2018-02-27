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

def sb_ext_square(cut_width, flexure_width, junction_length, edge_space, num_flex):
    """
    """
    import numpy as np
    a = cut_width; b = flexure_width; c = junction_length; d = edge_space
    sqrt2 = 2**0.5
    ax = a/sqrt2/2 # x displacement along diagonal cut
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


sb = sb_ext_square(cut_width=1, flexure_width=2, junction_length=5, edge_space=2, num_flex=4)
print(sb)