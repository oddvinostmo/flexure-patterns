# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:18:22 2018
Patterns generators
@author: oddvi
"""

import numpy as np
import os

from sys import version_info
from math import cos, sin, pi
from sys import path
path.append(r'C:\Users\oddvi\Google Drive\04 NTNU\Master thesis\Code')

#import shapely.geometry # polygon, box
#import shapely.affinity # translate, rotate, scale
#import shapely.ops # cascaded_union

from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate, scale, skew
from shapely.ops import cascaded_union, unary_union
from makeSVG import createSVG
from convert_graphics import batch_svg_to_pdf



class GeneratingRegion(): # shapely.geometry.Polygon
    def __init__(self, flexure_type, wallpaper_group):
#        super(self.__class__, self).__init__()
        self.flexure_type = flexure_type
        self.wallpaper_group = wallpaper_group
        self.generating_region = None
        self.para = dict()
        self.set_para() # also makes generating Region
#        self.make_generating_region()
        
    def __repr__(self):
        return '%s' % (self.__class__)
                
    def __str__(self):
        return ('Generating region type {0}'
                'flexure for wallapaper group {1}'
                ).format(self.flexure_type, self.wallpaper_group)
    
    def plot(self):
        self.plot_polygon(self.generating_region)
        
    def set_para(self, **kwargs):
        # Set values if none exists
        if self.para == dict():
            if 'switchback' in self.flexure_type and self.wallpaper_group == 'cmm':
                self.para = {'flexure_length':2.0, 'flexure_width':1.0, 
                             'stem_width':1.0, 'cut_width':1.0, 'num_turns':1}
                
            elif 'switchback' in self.flexure_type and self.wallpaper_group != 'cmm':
                self.para = {'flexure_width':1.0, 'cut_width':1.0, 'num_flex':1, 
                             'junction_length':2.0, 'side_cut':'default', 
                             'stem_width':'default'}
            
            elif 'let' in self.flexure_type and self.wallpaper_group != 'cmm':
                self.para = {'flexure_width':1.0, 'stem_width':1.0, 'cut_width':1.0, 
                             'junction_length':2.0,  'num_flex':1, 
                             'inside_start':True, 'edge_space':1.0}

            elif 'let' in self.flexure_type and self.wallpaper_group == 'cmm':
                self.para = {'flexure_width':1.0, 'flexure_length':1.0, 
                             'stem_length':1.0, 'stem_width':1.0}
            
            elif 'swastika' in self.flexure_type:
                self.para= {'flexure_length':1.0, 'flexure_width':1.0}
            elif 'coil' in self.flexure_type:
                self.para = {'cut_width':1.0, 'flexure_width':1.0}

            elif 'ydx' in self.flexure_type:
                self.para = {'flexure_length':1.0, 'flexure_width':1.0, 
                             'cut_width':1.0, 'solid_width':1.0, 'thetaDeg':45.0}

            elif 'solid' in self.flexure_type:
                self.para = {'width':1.0, 'height':1.0}
        
        # changes values if given
        if version_info[0] <3: # checks for python 2.7
            iterator = kwargs.iteritems()
        else:
            iterator = kwargs.items()
        for key, val in iterator:
            self.para[key] = float(val)
        # Remake generating region
        self.make_generating_region()
    
    def make_generating_region(self):
        """Finds the right generator by comparing the flexure type 
        and wallpaper_group
        """
        if self.flexure_type == 'ydx':
            self.make_ydx_gen_reg()
        
        elif 'let' in self.flexure_type:
            if self.wallpaper_group == 'p4m':
                self.make_square_let_gen_reg()
            
            elif 'hex' in self.flexure_type: # can add... self.wallpaper_group == 'p6m' and 
                self.make_hexagonal_let_gen_reg()
                
            elif 'tri' in self.flexure_type:
                self.generating_region = self.make_triangular_let_gen_reg(self.para)
            
            elif self.wallpaper_group == 'cmm':
                self.generating_region = self.make_torsion_flexure(self.para)
                
        elif 'switchback' in self.flexure_type:            
            if self.wallpaper_group == 'p4':
                self.generating_region = self.make_square_switchback_gen_reg(self.para)
                
            elif self.wallpaper_group == 'p4g':
                self.generating_region = self.make_square_p4g_switchback_gen_reg()
            
            elif 'tri' in self.flexure_type and self.wallpaper_group == 'p6':
                self.make_triangular_switchback_gen_reg()
            
            elif 'hex' in self.flexure_type and self.wallpaper_group == 'p6':
                pass# self.generating_region = self.make_hexagonal_switchback_gen_reg(self.para)
            
            elif self.wallpaper_group == 'cmm':
                self.make_rectangular_switchback_gen_reg()
        
        elif 'coil' in self.flexure_type:
                self.make_square_coil_gen_reg()
            
        elif 'swastika' in self.flexure_type:
                self.make_square_swastika()
                
        elif 'solid' in self.flexure_type:
            self.make_solid_square()
        
        elif 'test' in self.flexure_type and 'pm' in self.wallpaper_group:
            self.make_pm_test()
        elif 'test' in self.flexure_type and 'pg' in self.wallpaper_group:
            self.make_pg_test()
        elif 'test' in self.flexure_type and 'p2' in self.wallpaper_group:
            self.make_p2_test()        
        elif 'test' in self.flexure_type and 'p1' in self.wallpaper_group:
            self.make_p1_test()
            
    def make_p1_test(self):
        coords = []
        coords.append((-3.0,-3.5))
        coords.append((-2.0,-3.5))
        coords.append((-2.0,-2.5))
        coords.append((3,-2.5))
        coords.append((3,1.5))
        coords.append((4,1.5))
        coords.append((4,2.5))
        coords.append((2,2.5))
        coords.append((2,-1.5))
        coords.append((-2.0,-1.5))
        coords.append((-2.0,-0.5))
        coords.append((1.0,-0.5))
        coords.append((1.0,2.5))
        coords.append((-2.0,2.5))
        coords.append((-2.0,3.0))
        coords.append((-3.0,3.0))
        coords.append((-3.0,2.5))
        coords.append((-3.5,2.5))
        coords.append((-3.5,1.5))
        coords.append((0.0,1.5))
        coords.append((0.0,0.5))
        coords.append((-3.0,0.5))
        self.generating_region = Polygon(coords)
        
    def make_pm_test(self):
        coords = []
        coords.append((0,0))
        coords.append((0,4))
        coords.append((1,4))
        coords.append((4,2.5))
        coords.append((7,4))
        coords.append((8,4))
        coords.append((8,0))
        coords.append((7,0))
        coords.append((7,3))
        coords.append((4,1.5))
        coords.append((1,3))
        coords.append((1,0))
        self.generating_region = Polygon(coords)
        
    def make_pg_test(self):
        coords = []
        coords.append((0.5,0))
        coords.append((0.5,7))
        coords.append((1,7))
        coords.append((1,4))
        coords.append((3,6))
        coords.append((3,7))
        coords.append((4,7))
        coords.append((4,4))
        coords.append((6,2))
        coords.append((6,7))
        coords.append((6.5,7))
        coords.append((6.5,0))
        coords.append((6,0))
        coords.append((6,1))
        coords.append((4,3))
        coords.append((4,0))
        coords.append((3,0))
        coords.append((3,5))
        coords.append((1,3))
        coords.append((1,0))
        self.generating_region = Polygon(coords)        

    def make_p2_test(self):
        coords = []
        coords.append((0,0))
        coords.append((0,4))
        coords.append((1,4))
        coords.append((1,1))
        coords.append((5,4))
        coords.append((6,4))
        coords.append((6,0))
        coords.append((5,0))
        coords.append((5,3))
        coords.append((1,0))
        self.generating_region = Polygon(coords)
    
    @staticmethod
    def plot_polygon(polygon):
        """Plots a shapely polygon using the matplotlib library """
        import matplotlib.pyplot as plt
        text = ''
        if polygon.type == 'MultiPolygon': 
            text = (polygon.type)
        try:
            fig = plt.figure(1, figsize=(5,5), dpi=90)
            ax = fig.add_subplot(111)
            ax.set_title('Polygon')
            ax.axis('equal')
            x,y = polygon.exterior.coords.xy
            ax.plot(x,y,'blue')
            for interior in polygon.interiors:
                x,y = interior.coords.xy
                ax.plot(x, y, 'blue')
        except:
            text = 'Plotting failed of {0}'.format(polygon.type)
        print(text)
            
    @staticmethod
    def make_torsion_flexure(para_dict):
        """             __
         ______________|  | I stem_length(d)
        |   ______________| I flexure_width (c) # width
        |__|<--------->|   flexure_length (b)
        <--> stem_width (a)
        l1, l2, angle = 2*a+b, 2*c+d, 90
        """
        a = para_dict['stem_width']
        b = para_dict['flexure_length']
        c = para_dict['flexure_width']
        d = para_dict['stem_length']
        x1 = 0; y1 = c+d
        x2 = a+b; y2 = y1
        x3 = x2; y3 = c+2*d
        x4 = 2*a+b; y4 = y3
        x5 = x4; y5 = d
        x6 = a; y6 = y5
        x7 = a; y7 = 0
        x8 = 0; y8 = 0
        x = [x1,x2,x3,x4,x5,x6,x7,x8]
        y = [y1,y2,y3,y4,y5,y6,y7,y8]
        coords = list(zip(x,y))
        return Polygon(coords)
        
    def make_solid_square(self):
        self.generating_region = box(0, 0, self.para['width'], self.para['height'])
    
    def make_square_let_gen_reg(self):
        """
        Generation of 1/8 of square cyclic slits. Full cell is generated with p4m.
        Returns a exterior ring of coorinates
        l1, l2, angle = h, h, 45
        """
        a = self.para['cut_width']
        b = self.para['flexure_width']
        c = self.para['junction_length']
        d = self.para['edge_space']
        e = self.para['stem_width']
        num_flex = int(self.para['num_flex'])
        inside_start = self.para['inside_start']
        
        sqrt2 = 2**0.5
        ax = a*sqrt2/2.0 # x displacement along diagonal cut
        d = d*sqrt2
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
        self.generating_region = Polygon(coords)
    
    @staticmethod
    def make_triangular_let_gen_reg(para_dict):
        """
        Generates 1/12 of a LET that maps with the p6m group
        """
        a = para_dict['cut_width']
        b = para_dict['flexure_width']
        c = para_dict['junction_length']
        d = para_dict['edge_space']
        e = para_dict['stem_width']
        num_flex = int(para_dict['num_flex'])
        inside_start = para_dict['inside_start']
        sqrt3 = 3**0.5
        d = d*sqrt3
        dy = a+b # displacement y direction
        dx = sqrt3*dy # displacement x direction
        h0 = c+a/2.0 # height in triangle
        ax = a*sqrt3/2.0 # x displacement along diagonal cut
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
                vec_x = np.append(vec_x, [ax+d]) # 5, add/remove +l1*sqrt3 for drafted corner
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
        return Polygon(list(zip(x,y)))
    
    def make_hexagonal_let_gen_reg(self): # BUG !!! ??
        """
        Generation of 1/8 of hexagonal cyclic slits. Full cell is generated with p6m.
        s = scaled, r = rotated, m = mirrored, t = translated
        dimensions are scaled - fix!!
        """
        para_dict = dict(self.para)
        para_dict['edge_space'] *= 3
        para_dict['stem_width'] *= 3

        generating_unit = self.make_triangular_let_gen_reg(para_dict) # have to add a edge cut paramter...
        p6m_sm = scale(geom=generating_unit, xfact=(-1/3), yfact=1, zfact=1, origin='center')
        p6m_smr = rotate(geom=p6m_sm, angle=90, origin='center')
        xmin, ymin, xmax, ymax = p6m_smr.bounds  
        p6m_smrt = translate(geom=p6m_smr, xoff = -xmin, yoff=-ymin)
        self.generating_region = p6m_smrt
    
    def make_rectangular_switchback_gen_reg(self):
        """" 
        NB! num_turns >= 1
        """
        # Make dictionary to pass into make_torsion_flexure - start segment
        para_dict = {'flexure_width':self.para['flexure_width'], 
                     'flexure_length':self.para['flexure_length'], 
                     'stem_length':self.para['cut_width']/2.0, # cut_width/2
                     'stem_width':self.para['flexure_width']} # stem_width 
        # Make dictionary to pass into make_torsion_flexure - middle segment
        para_dict_middle = dict(para_dict)
        para_dict_middle['flexure_length'] = para_dict['flexure_length'] - self.para['cut_width']/2.0
        # Shortcut variables
        num_turns = int(self.para['num_turns'])
        dy = self.para['cut_width'] + self.para['flexure_width']
        dx = self.para['cut_width']/2.0
        # Switchback list
        swichbacks = []
        # Make first segment
        start_segment = self.make_torsion_flexure(para_dict)
        swichbacks.append(start_segment)
        # Make middle segment
        middle_segment = self.make_torsion_flexure(para_dict_middle)
        middle_segment_mirror = scale(middle_segment, xfact=-1, yfact=1, origin='center')
        # Adds number of desired turns
        for n in range(0, num_turns):
            if n == 0:
                swichbacks.append(translate(middle_segment_mirror, xoff=dx, yoff=dy))
            else:
                swichbacks.append(translate(middle_segment, xoff=dx, yoff=dy*(2*n)))
                swichbacks.append(translate(middle_segment_mirror, xoff=dx, yoff=dy*(2*n+1)))
        # Make last segment
        end_segment = translate(start_segment, xoff=dx, yoff=dy*(num_turns*2))
        swichbacks.append(end_segment)
        self.generating_region = cascaded_union(swichbacks)
        
        
    def make_triangular_switchback_gen_reg(self): # BUG! not finished!
        """ not ideal way of sorting the 'default' options...
        """
        para_dict = dict(self.para)
        # Fist initialize to get height
        sq_gen_reg = self.make_square_switchback_gen_reg(para_dict)
        xmin, ymin, xmax, ymax = sq_gen_reg.bounds
        # Calculate xscale from height
        xscale = 3**0.5#*(ymax-ymin)
#        if para_dict['side_cut'] == 'default':
#            para_dict['side_cut'] = para_dict['cut_width']*3**0.5/2*(ymax-ymin)*xscale
#        else:
#            para_dict['side_cut'] *= xscale
#        if para_dict['stem_width'] == 'default':
#            para_dict['stem_width'] = para_dict['flexure_width']*3**0.5/2*(ymax-ymin)*xscale
#        else:
#            para_dict['stem_width'] *= xscale
        # Make new switchback with adjusted lengths
        gen_reg = self.make_square_switchback_gen_reg(para_dict)
        self.generating_region = scale(gen_reg, xfact=xscale)
#         = translate(gen_reg, xoff=-xmin, yoff=-ymin)

    @staticmethod
    def make_square_switchback_gen_reg(para_dict):
        a = para_dict['cut_width']
        ax = a*(2**0.5)/2 if para_dict['side_cut'] == 'default' else para_dict['side_cut']  
        b = para_dict['flexure_width']
        c = para_dict['junction_length']
        d = b*2**0.5 if para_dict['stem_width'] == 'default' else para_dict['stem_width']
        num_flex = int(para_dict['num_flex'])
        # x distance along diagonal cut
        
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
        return Polygon(list(zip(x,y)))

    def make_square_p4g_switchback_gen_reg(self):
        para_dict = dict(self.para)
        # Fist initialize to get height
        sq_gen_reg = self.make_square_switchback_gen_reg(para_dict)
        xmin, ymin, xmax, ymax = sq_gen_reg.bounds
        intersection_box = box(xmin, ymin+self.para['flexure_width']/2. + self.para['cut_width']/2., xmax, ymax)
        return sq_gen_reg.intersection(intersection_box)
        
    def make_ydx_gen_reg(self):
        """
        Full unit generated through pmm
        l1, l2, angle = w, h, 90
        """
        import math
        a = self.para['solid_width']
        b = self.para['flexure_length']
        c = self.para['flexure_width']
        d = self.para['cut_width']
        thetaDeg = self.para['thetaDeg']
        
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
        self.generating_region = Polygon(coords)
    
    def make_square_swastika(self):
        a = self.para['flexure_length'] #flexure_length_1
#        b = self.para['flexure_length'] #flexure_length_2
        c = self.para['flexure_width'] #flexure_width
        b = a+c/2+(a-c/2)/2
        coords = list()
        coords.append((0,0)) # 0 
        coords.append((0,a+c)) # 1
        coords.append((b+c,a+c)) # 2
        coords.append((b+c,a)) # 3
        coords.append((c,a)) # 4
        coords.append((c,0)) # 5
        polygon = Polygon(coords)
        polygons = []
        for i in range(4):
            polygons.append(rotate(polygon, angle=90*i, origin=(c/2,0)))
        self.generating_region = scale(unary_union(polygons), xfact=-1)
        
#        swastikas = [swastika]
#        xmin, ymin, xmax, ymax = swastika.bounds
#        swastikas.append(scale(swastika, xfact=-1, origin=(xmax,ymin)))
#        swastikas.append(scale(swastika, yfact=-1, origin=(xmax,ymin)))
#        swastikas.append(scale(swastika, xfact=-1, yfact=-1, origin=(xmax,ymin)))
#        swastika_cmm = unary_union(swastikas)
    
    def make_square_coil_gen_reg(self):
        a = float(self.para['cut_width'])
        b = float(self.para['flexure_width'])
        coords = list()
#        coords.append((0/2*a+0*b, 0/2*a+0*b)) #0
        coords.append((0.0, 0.0))
        coords.append((1.0/2*a+1*b, 0.0/2*a+0*b)) #1 
        coords.append((1.0/2*a+1*b, 1.0/2*a+0*b)) #2
        coords.append((9.0/2*a+5*b, 1.0/2*a+0*b)) #3
        coords.append((9/2*a+5*b, 7/2*a+4*b)) #4
        coords.append((3/2*a+1*b, 7/2*a+4*b)) #5
        coords.append((3/2*a+1*b, 5/2*a+2*b)) #6
        coords.append((7/2*a+3*b, 5/2*a+2*b)) #7
        coords.append((7/2*a+3*b, 3/2*a+2*b)) #8
        coords.append((1/2*a+1*b, 3/2*a+2*b)) #9
        coords.append((1/2*a+1*b, 9/2*a+4*b)) #10
        coords.append((5*a+5*b,   9/2*a+4*b)) #11
        coords.append((5*a+5*b,   5*a+5*b)) #12
        # Half way
        coords.append((9/2*a+4*b, 5*a+5*b)) #13
        coords.append((9/2*a+4*b, 9/2*a+5*b)) #14
        coords.append((1/2*a,     9/2*a+5*b)) # 15
        coords.append((1/2*a,     3/2*a+b)) # 16
        coords.append((7/2*a+4*b, 3/2*a+1*b)) #17
        coords.append((7/2*a+4*b, 5/2*a+3*b)) #18
        coords.append((3/2*a+2*b, 5/2*a+3*b)) # 19
        coords.append((3/2*a+2*b, 7/2*a+3*b)) #20
        coords.append((9/2*a+4*b, 7/2*a+3*b)) # 21
        coords.append((9/2*a+4*b, 1/2*a+1*b)) # 22
        coords.append((0,         1/2*a+1*b)) # 23
        self.generating_region = Polygon(coords)


class Unit(GeneratingRegion):
    def __init__(self, flexure_type, wallpaper_group):
#        super(Unit, self).__init__(flexure_type, wallpaper_group) # for python 3.
        GeneratingRegion.__init__(self, flexure_type, wallpaper_group) # for python 2.x
        self.unit = None
        self.flexure_pattern = None
        self.name = self.flexure_type +'_'+ self.wallpaper_group
        self.make_unit()
        self.get_dimensions()
        self.get_center()
        self.get_corners()
    
    def __repr__(self):
        return '%s' % (self.__class__)
    
    def __str__(self):
        return 'A unit object'

    def plot(self):
        self.plot_polygon(self.unit)

    def make_valid(self):
#        if '3' in self.wallpaper_group or '6' in self.wallpaper_group:
        self.unit = self.unit.buffer(0.0001, join_style=2)
        if self.unit.type == 'MultiPolygon':
            print('the unit is still a MultiPolygon!')
        
    def make_unit(self):
        if 'special' in self.wallpaper_group:
            self.special()
        elif 'test' in self.flexure_type:
            self.unit = self.generating_region
        elif self.wallpaper_group == 'pmm' or self.wallpaper_group == 'cmm':
            self.pmm()
        elif self.wallpaper_group == 'p4':
            self.p4()
        elif self.wallpaper_group == 'p4m':
            self.p4m()
        elif self.wallpaper_group == 'p4g':
            self.p4g()
        elif self.wallpaper_group == 'pm':
            self.pm()
        elif self.wallpaper_group == 'p6':
            self.p6()
        elif self.wallpaper_group == 'p6m':
            self.p6m()            
    
    def get_dimensions(self):
        xmin, ymin, xmax, ymax = self.unit.bounds
        if '3' in self.wallpaper_group or '6' in self.wallpaper_group:
            self.angle = 60
            self.angle_rad = self.angle*pi/180
            self.l1 = (ymax-ymin)/sin(self.angle_rad)
            self.l2 = self.l1
        else:
            self.angle = 90
            self.angle_rad = self.angle*pi/180
            self.l1 = xmax-xmin
            self.l2 = ymax-ymin

    def get_center(self):
        xmin, ymin, xmax, ymax = self.unit.bounds
        self.center = ((xmin+xmax)/2, (ymin+ymax)/2)
        
    def get_corners(self):
        self.get_center()
        d1 = (self.l1 + self.l2*cos(self.angle_rad))/2
        d2 = self.l2*sin(self.angle_rad)/2
        cx, cy = [self.center[0], self.center[1]]
        self.unit_coords = [(cx-d1, cy-d2), (cx+self.l1-d1, cy-d2), 
                            (cx+d1, cy+d2), (cx-self.l1+d1, cy+d2)]
        
    def update(self):
        self.get_center()
        self.get_corners()
        
    def set_name(self, name):
        self.name = self.flexure_type + '_' + self.wallpaper_group + '-' + str(name)
    
    def set_unit_para(self, **kwargs):
        self.set_para(**kwargs)
        self.make_unit()
        self.get_dimensions()
        self.get_center()
        self.get_corners()
        self.get_lattice_type()
        
    def translate_unit(self, xoff, yoff):
        # Used by abaqus
        self.unit = translate(self.unit, xoff=xoff, yoff=yoff)
        self.update() 
        
    def get_lattice_type(self):
        if self.angle > 90:
            print('Invalid too big angle')
            return
        if self.angle == 90  and self.l1 == self.l2:
            self.lattice_type = 'square'
        elif self.angle == 90 and self.l1 != self.l2:
            self.lattice_type = 'rectangle'
        elif self.angle != 90 and self.l1 == self.l2:
            self.lattice_type = 'rhombe'
        else:
            self.lattice_type = 'parallelogram'
        
    def pmm(self):
        generating_unit = self.generating_region
        [xmin, ymin, xmax, ymax] = generating_unit.bounds
        mirrored_x = scale(generating_unit, xfact=-1, yfact=1, origin=(xmin,ymin))
        mirrored_y = scale(generating_unit, xfact=1, yfact=-1, origin=(xmin,ymin))
        mirrored_xy = scale(generating_unit, xfact=-1, yfact=-1, origin=(xmin,ymin))
        unit_cell = cascaded_union([generating_unit, mirrored_x, mirrored_y, mirrored_xy])
        self.unit = unit_cell
#        self.make_valid()
        
    def p4m(self):
        generating_unit = self.generating_region
        [xmin, ymin, xmax, ymax] = generating_unit.bounds
        mirrored_y = scale(generating_unit, xfact=-1,yfact=1,origin=(xmax,ymax))
        shapes = []
        for i in range(0,4):
            shapes.append(rotate(generating_unit,angle=(90*i), origin=(xmax,ymax)))
            shapes.append(rotate(mirrored_y,angle=(90*i), origin=(xmax,ymax)))
        self.unit = cascaded_union(shapes)
        self.make_valid()
       
    def p4(self):
        generating_unit = self.generating_region
        xmin, ymin, xmax, ymax = generating_unit.bounds
        unit_list = [generating_unit]
        # exception for the triangular generating region (switchback cases)
        point = ((xmax+xmin)/2, ymax) if self.flexure_type == 'switchback' else (xmin, ymin)
        for i in range(1,4):
            unit_list.append(rotate(generating_unit, angle=i*90, origin=point))
        self.unit = unary_union(unit_list) # unary_union(unit_list)
        self.make_valid()
        
    def p4g(self):
        """Only implemented to work with square switchback """
        generating_region = self.generating_region
        # rotate algorithm to get absolute center
        generating_region = rotate(generating_region, angle=135)
        centerx, centery, xmax, ymax = generating_region.bounds
        generating_region = rotate(generating_region, angle=-135, origin=(centerx, centery))
        xmin, ymin, xmax, ymax = generating_region.bounds
        # mirror along x
        polygon_1 = scale(generating_region, xfact=1, yfact=-1, origin=(xmin, ymin))
        polygon_2 = unary_union([generating_region, polygon_1])
        polygons = []
        # 4-fold rotation
        for i in range(4):
            polygons.append(rotate(polygon_2, angle=90*i, origin=(centerx, centery)))
        # union and rotate
        self.unit = rotate(unary_union(polygons), angle=45)
        self.make_valid()
        
                
    def pm(self):
        generating_unit = self.generating_region
        [xmin, ymin, xmax, ymax] = generating_unit.bounds
        mirrored_y = scale(generating_unit, xfact=1,yfact=-1,origin=(ymin,xmin))
        self.unit = cascaded_union([generating_unit,mirrored_y])
#        self.make_valid()
        
    def p6(self):
        xmin, ymin, xmax, ymax = self.generating_region.bounds
        dy = ymax-ymin
        cx = (xmax+xmin)/2 # center x-direction
        shapes = []
        for i in range(3):
            shapes.append(rotate(self.generating_region, angle=120*i, origin=(cx, ymax)))
        triangle = unary_union(shapes)
        triangle_rot = rotate(triangle, angle=180, origin=(cx+dy/(3**0.5)+1.5,ymax+0.5*dy))
#        triangle_trans_rot = translate(triangle_rot, xoff=2/3**0.5*dy, yoff=dy)
#        triangle_rot = rotate(triangle, angle=180, origin=(cx+dy, ymax+dy*3**0.5/2))# origin=(ymin + 3**0.5/2*(ymax+ymin)/2, ymin))
        self.unit = unary_union([triangle, triangle_rot])
#        self.unit = triangle_trans_rot
#        self.unit = triangle_rot
#        self.unit = triangle
        self.make_valid()
        
    def p6m(self): # BUG!
        """ Notes: works only for 120 deg flexure, one rotation center is a bit off...  
        """
        generating_region = self.generating_region
        xmin, ymin, xmax, ymax = generating_region.bounds
        mirrored_y = scale(generating_region, xfact=-1,yfact=1, origin=(xmax,ymax))
        shapes = []
        for i in range(0,4):
            shapes.append(rotate(generating_region,angle=(120*i), origin=(xmax,ymax)))
            shapes.append(rotate(mirrored_y,angle=(120*i), origin=(xmax,ymax)))
        triangle = unary_union(shapes).buffer(0.01, resolution=1, join_style=2)
        rot_center = (xmax + 3**0.5*(ymax-ymin), ymin)
        triangle_rot = rotate(triangle, angle=300, origin=rot_center)
#        self.unit = triangle.union(triangle_rot)
        self.unit = unary_union([triangle, triangle_rot])
#        self.make_valid()
        
    def special(self):
        p = self.generating_region
        xmin, ymin, xmax, ymax = p.bounds
        mx = scale(p, xfact=-1, origin=(xmax, ymin))
        my = scale(p, yfact= -1, origin=(xmax, ymin))
        mxy = scale(p, xfact=-1, yfact=-1, origin=(xmax, ymin))
        s1 = unary_union([p, mx, my, mxy])
        s1r2 = rotate(s1, angle=240, origin=(xmax, ymax))
        pr = rotate(p, angle=180, origin=(xmax/2, 3**0.5/2*xmax))
        cell_quart = unary_union([p, pr, s1r2])
        cell_quart_1 = scale(cell_quart, xfact=-1, origin=(xmax, ymin))
        cell_quart_2 = scale(cell_quart, yfact=-1, origin=(xmax, ymin))
        cell_quart_3 = scale(cell_quart, xfact=-1, yfact=-1, origin=(xmax, ymin))
        cell = unary_union([cell_quart, cell_quart_1, cell_quart_2, cell_quart_3])
        self.unit = cell
        
    def make_pattern(self, ncell_x, ncell_y):
        """
        Creates a nx x ny surface of units cells. For hexagonal units, hex_cell = True
        """
        duplicates = []
        for i in range(ncell_y):
            for j in range(ncell_x):
                duplicates.append(translate(self.unit, 
                        xoff=(self.l1*j + cos(self.angle_rad)*self.l2*i), 
                        yoff=(self.l2 * sin(self.angle_rad) * i)))
        self.flexure_pattern = unary_union(duplicates).buffer(0.00001, resolution=1, cap_style=2)

def batch_save_svg(unit):
    width = 100
    name = unit.name
    nx = 4
    # Calculate close to square
    ny = int(unit.l1*nx/unit.l2)
    # Tile the patten
    unit.make_pattern(nx, ny)
    # Save
    createSVG(unit.unit, name+'_unit.svg', width)
    createSVG(unit.flexure_pattern, name+'_flex_pat.svg', width)
    createSVG(unit.generating_region, name+'_gen_reg.svg', width)




if __name__ == '__main__':
    """
    flexure_types implemented: let, switchback, coil, solid
    wallper_generators implemented: p4, p6m
    special combinations: switchback-1 and -2 with p6, let-1 and -2 with p6m
    let cmm
    let p4m
    let-tri p6m (special case)
    let-hex p6m (special case)
    switchback cmm
    switchback p4 (special case)
    switchback-tri p6 (special case)
    switchback-hex p6 (special case)
    wip switchback p4g
    ydx cmm
    coil p4
    coil cmm
    test p2
    test pm
    test pg
    
    """
    flexure_type = 'ydx'
    wallpaper_group = 'cmm'
    gen_reg = GeneratingRegion(flexure_type, wallpaper_group)
    unit = Unit(flexure_type, wallpaper_group)
    unit.set_unit_para(thetaDeg=80, flexure_length= 5, solid_wifth = 5, cut_width = 0.5)
    batch_save_svg(unit)
    batch_svg_to_pdf(os.getcwd()) 
    unit.plot()
