# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:18:22 2018
Patterns generators
@author: oddvi
"""

import numpy as np
from sys import version_info
from math import cos, sin, pi

#import shapely.geometry # polygon, box
#import shapely.affinity # translate, rotate, scale
#import shapely.ops # cascaded_union

from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate, scale, skew
from shapely.ops import cascaded_union, unary_union
from makeSVG import createSVG


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
                             'junction_length':2.0,  'num_flex':2, 
                             'inside_start':True, 'edge_space':1.0}

            elif 'let' in self.flexure_type and self.wallpaper_group == 'cmm':
                self.para = {'flexure_width':1.0, 'flexure_length':1.0, 
                             'stem_length':1.0, 'stem_width':1.0}
            
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
            
            elif self.wallpaper_group == 'p6m' and 'hex' in self.flexure_type:
                self.make_hexagonal_let_gen_reg()
                
            elif self.wallpaper_group == 'p6m' and 'tri' in self.flexure_type:
                self.generating_region = self.make_triangular_let_gen_reg(self.para)
            
            elif self.wallpaper_group == 'cmm':
                self.generating_region = self.make_torsion_flexure(self.para)
                
        elif 'switchback' in self.flexure_type:            
            if self.wallpaper_group == 'p4':
                self.generating_region = self.make_square_switchback_gen_reg(self.para)
            
            elif 'tri' in self.flexure_type and self.wallpaper_group == 'p6':
                self.generating_region = self.make_triangular_switchback_gen_reg()
            
            elif 'hex' in self.flexure_type and self.wallpaper_group == 'p6':
                pass# self.generating_region = self.make_hexagonal_switchback_gen_reg(self.para)
            
            elif self.wallpaper_group == 'cmm':
                self.make_rectangular_switchback_gen_reg()
        
        elif 'coil' in self.flexure_type:
            if self.wallpaper_group == 'cmm' or self.wallpaper_group == 'p4':
                self.make_square_coil_gen_reg()
                
        elif 'solid' in self.flexure_type:
            self.make_solid_square()
            
#        elif 'special1' in self.flexure_type:
#            self.make_special1()
    
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
    
#    def make_hexagonal_let_gen_reg(self): # BUG !!! ??
#        """
#        Generation of 1/8 of hexagonal cyclic slits. Full cell is generated with p6m.
#        s = scaled, r = rotated, m = mirrored, t = translated
#        dimensions are scaled - fix!!
#        """
#        para_dict = dict(self.para)
#        para_dict['edge_space'] *= 3
#        para_dict['stem_width'] *= 3
#
#        generating_unit = self.make_triangular_let_gen_reg(para_dict) # have to add a edge cut paramter...
#        p6m_sm = scale(geom=generating_unit, xfact=(-1/3), yfact=1, zfact=1, origin='center')
#        p6m_smr = rotate(geom=p6m_sm, angle=90, origin='center')
#        xmin, ymin, xmax, ymax = p6m_smr.bounds  
#        p6m_smrt = translate(geom=p6m_smr, xoff = -xmin, yoff=-ymin)
#        self.generating_region = p6m_smrt
    
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
        xscale = 3**0.5*(ymax-ymin)/2
        if para_dict['side_cut'] == 'default':
            para_dict['side_cut'] = para_dict['cut_width']*3**0.5/2*(ymax-ymin)*xscale
        else:
            para_dict['side_cut'] *= xscale
        if para_dict['stem_width'] == 'default':
            para_dict['stem_width'] = para_dict['flexure_width']*3**0.5/2*(ymax-ymin)*xscale
        else:
            para_dict['stem_width'] *= xscale
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
        
#        coords_1 = np.array([])
#        for i in coords:
#            coords_1 = np.append(coords_1, i)
#        print(coords_1)
        x = [c[0] for c in coords]
        y = [c[1] for c in coords]
        coordinates = list(zip(x,y))
#        coordinates = [(x[i], y[i]) for i in range(len(x))]
#           # RIP        
#        line_1 = LineString(coords)
#        line_2 = rotate(line_1, angle=180, origin='center')
#        coords_merge = list(line_1.coords)+list(line_2.coords)
        self.generating_region = Polygon(coordinates)


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
        if self.wallpaper_group == 'pmm' or self.wallpaper_group == 'cmm':
            self.pmm()
        elif self.wallpaper_group == 'p4m':
            self.p4m()
        elif self.wallpaper_group == 'p4':
            self.p4()
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
                
    def pm(self):
        generating_unit = self.generating_region
        [xmin, ymin, xmax, ymax] = generating_unit.bounds
        mirrored_y = scale(generating_unit, xfact=1,yfact=-1,origin=(ymin,xmin))
        self.unit = cascaded_union([generating_unit,mirrored_y])
#        self.make_valid()
        
    def p6(self):
        xmin, ymin, xmax, ymax = self.generating_region.bounds
        shapes = []
        for i in range(3):
            shapes.append(rotate(self.generating_region, angle=120*i, origin=((xmax+xmin)/2, ymax)))
        triangle = unary_union(shapes)
        trinagle_rot = rotate(triangle, angle=300, origin=(ymin + 3**0.5/2*(ymax-ymin)/2, ymin))
        self.unit = unary_union(triangle, trinagle_rot)
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
        triangle = unary_union(shapes)
        rot_center = (xmax + 3**0.5*(ymax-ymin), ymin)
        triangle_rot = rotate(triangle, angle=300, origin=rot_center)
        self.unit = unary_union([triangle, triangle_rot])
        self.make_valid()
        
    def tile(self, ncell_x, ncell_y):
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
    # Tile the pattern
    unit.tile(nx, ny)
    # Save
    createSVG(unit.unit, name+'_unit.svg', width)
    createSVG(unit.flexure_pattern, name+'_flex_pat.svg', width)
    createSVG(unit.generating_region, name+'_gen_reg.svg', width)

#    
#class printSVG():
#    def __init__(self):
        


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
    ydx cmm
    
    """
    flexure_type = 'let-tri'
    wallpaper_group = 'p6m'
    gen_reg = GeneratingRegion(flexure_type, wallpaper_group)
#    unit = Unit(flexure_type, wallpaper_group)
#    unit.set_unit_para(num_flex=2, cut_width=1)
#    batch_save_svg(unit)
     
     
     
     



























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


def make_hexagonal_let_unit(cut_width, flexure_width, junction_length, 
                       edge_space, stem_width, num_flex, inside_start):
    """
    Real input length is not accurate, multiply something with 3 to make accurate measures...
    """
    edge_space *= 3
    stem_width *= 3
    generating_unit = make_hexagonal_let_gen_reg(cut_width, flexure_width, junction_length, edge_space, stem_width, num_flex, inside_start)
    xmin, ymin, xmax, ymax = generating_unit.bounds 
    gen_mx = scale(geom=generating_unit, xfact=1.0, yfact=-1.0, zfact=1.0, origin=(xmax,ymin))
    gen_my = scale(geom=generating_unit, xfact=-1.0, yfact=1.0, zfact=1.0, origin=(xmax,ymin))
    gen_mxy = scale(geom=generating_unit, xfact=-1.0, yfact=-1.0, zfact=1.0, origin=(xmax,ymin))
    mirror = cascaded_union([gen_mx, gen_mxy])
    xmin, ymin, xmax, ymax = mirror.bounds
    mirror_rot1 = rotate(geom=mirror, angle=60, origin=(xmin,ymax))
    mirror_rot2 = rotate(geom=mirror, angle=-60, origin=(xmax,ymax))
    one_half = cascaded_union([mirror_rot1, mirror_rot2, generating_unit, gen_my])
    xmin, ymin, xmax, ymax = one_half.bounds
    other_half = rotate(geom=one_half, angle=-60, origin=(xmax,ymin))
    unit = cascaded_union([one_half, other_half])
    return unit
    
"""
Swichback generator
"""

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
    return scale(geom=sb_square_mod, xfact=xfact)

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
    return scale(geom=sb_square_mod, xfact=xfact)




"""
Master generator
Combines previous functions to create a unit of the desired pattern
"""


def make_square_let_unit(cut_width,flexure_width,junction_length,edge_space,stem_width,num_flex,inside_start):
    sq_cyclic_slit_gen = make_square_let_gen_reg(cut_width,flexure_width,junction_length,edge_space,stem_width,num_flex,inside_start)
    sq_cyclic_slit_unit = make_p4m_unit(sq_cyclic_slit_gen)
    return sq_cyclic_slit_unit

def make_rectangular_switchback_unit(num_turns, width_stem, length_flex, cut_width, width_flex):
    generating_region = make_rectangular_switchback_gen_reg(num_turns, width_stem, length_flex, cut_width, width_flex)
    return make_pmm_unit(generating_region)

#def make_square_coil_unit(cut_width, flexure_width):
##    gen_reg = make_square_coil_gen_reg(cut_width=cut_width, flexure_width=flexure_width)
##    l1 = l2 = 5*cut_width+5*flexure_width
##    angle = 90
##    unit = make_p4_unit(gen_reg)
#    x=[0.0, 0.0, 8.5, 8.5, 3.5, 3.5, 7.5, 7.5, 0.5, 0.5, 8.5, 8.5,  8.5,  3.5,  3.5,  4.5,  4.5,  7.5,  7.5,  0.5,  0.5,  0.0,  0.0,  1.5,  1.5,  6.5,  6.5,  5.5,  5.5,  2.5,  2.5,  9.5,  9.5,  10.0, 18.5, 18.5, 13.5, 13.5, 17.5, 17.5, 10.5, 10.5, 18.5, 18.5, 20.0, 20.0, 11.5, 11.5, 16.5, 16.5, 12.5, 12.5, 19.5, 19.5, 11.5, 11.5, 11.5, 16.5, 16.5, 15.5, 15.5, 12.5, 12.5, 19.5, 19.5, 20.0, 20.0, 18.5, 18.5, 13.5, 13.5, 14.5, 14.5, 17.5, 17.5, 10.5, 10.5, 10.0, 1.5, 1.5, 6.5, 6.5, 2.5, 2.5, 9.5, 9.5, 1.5, 1.5]
#    y=[0.0, 1.5, 1.5, 6.5, 6.5, 5.5, 5.5, 2.5, 2.5, 9.5, 9.5, 10.0, 18.5, 18.5, 13.5, 13.5, 17.5, 17.5, 10.5, 10.5, 18.5, 18.5, 20.0, 20.0, 11.5, 11.5, 16.5, 16.5, 12.5, 12.5, 19.5, 19.5, 11.5, 11.5, 11.5, 16.5, 16.5, 15.5, 15.5, 12.5, 12.5, 19.5, 19.5, 20.0, 20.0, 18.5, 18.5, 13.5, 13.5, 14.5, 14.5, 17.5, 17.5, 10.5, 10.5, 10.0, 1.5,  1.5,  6.5,  6.5,  2.5,  2.5,  9.5,  9.5,  1.5,  1.5,  0.0,  0.0,  8.5,  8.5,  3.5,  3.5,  7.5,  7.5,  0.5,  0.5,  8.5,  8.5,  8.5, 3.5, 3.5, 4.5, 4.5, 7.5, 7.5, 0.5, 0.5, 0.0]
#    coord = [(x,y) for x,y in zip(x,y)]
#    unit = Polygon(coord)
#    return unit


# Tiles

def make_triangular_switchback_tile(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    sb_third = make_triangular_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut)
    xmin, ymin, xmax, ycoord = sb_third.bounds
    xcoord = (xmax+xmin)/2
    sb_third_r120 = rotate(geom=sb_third, angle=120, origin=(xcoord,ycoord))
    sb_third_r240 = rotate(geom=sb_third, angle=240, origin=(xcoord,ycoord))
    return cascaded_union([sb_third, sb_third_r120, sb_third_r240])

def make_square_switchback_tile(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    sb_quart = make_square_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut)
    xmin, ymin, xmax, ycoord = sb_quart.bounds # ycoord is ymax
    xcoord = (xmax+xmin)/2
    sb_quart_r90 = rotate(geom=sb_quart, angle=90, origin=(xcoord,ycoord))
    sb_quart_r180 = rotate(geom=sb_quart, angle=180, origin=(xcoord,ycoord))
    sb_quart_r270 = rotate(geom=sb_quart, angle=270, origin=(xcoord,ycoord))
    return cascaded_union([sb_quart, sb_quart_r90, sb_quart_r180, sb_quart_r270])

def make_hexagonal_switchback_tile(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut='default'):
    sb_sixt = make_hexagonal_switchback_gen_reg(cut_width, flexure_width, junction_length, edge_space, num_flex, side_cut)
    xmin, ymin, xmax, ycoord = sb_sixt.bounds
    xcoord = (xmax+xmin)/2
    sb_rotated = [sb_sixt]
    for n in range(6):
        sb_rotated.append(rotate(geom=sb_sixt, angle=60*(1*n), origin=(xcoord,ycoord)))
    return cascaded_union(sb_rotated)


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
    LET_skew = skew(generating_unit, xs=0, ys=skew_angle, origin=(0,0))
    LET_skew_mirror = scale(LET_skew, xfact=-1, yfact=1, origin=(0,0))
    unit_cell = cascaded_union([LET_skew, LET_skew_mirror])
    unit_cell.buffer(0)
    # Maps the pattern
    [xmin, ymin, xmax, ymax] = generating_unit.bounds
    dx = (xmax-xmin)*2
    dy = ymax-ymin
    cell_duplicates = []
    for i in range(ncell_x):
        for j in range(ncell_y):
            cell_duplicates.append(translate(unit_cell, xoff=(dx*i), yoff=(dy*j)))
    surface_polygon = cascaded_union(cell_duplicates)    
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
    unit_cell = skew(generating_unit, xs=skew_angle, ys=0, origin=(0,0))
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
            cell_duplicates.append(translate(unit_cell, xoff=(x_off+dx*j), yoff=(dy*i)))
    surface_polygon = cascaded_union(cell_duplicates)    
    return surface_polygon



"""
Test calls for functions can be done by uncommenting the following lines
"""
if __name__ == '__main__':
    pass
##    Generators
#    plotPolygon(make_rectangular_switchback_gen_reg(num_turns=1, width_stem=1, length_flex=10, cut_width=1, width_flex=2))
#    
#    #Units
#    plotPolygon(make_ydx_unit(solid_width=1, flexure_length=5, flexure_width=1, cut_width = 0.5 ,thetaDeg=45))
#    
#    # LET
#    plotPolygon(make_outside_let(width_stem=1,length_flex=1,height_stem=1,width_flex=1))
#    plotPolygon(make_inside_let(width_stem=1,length_flex=1,height_stem=1,width_flex=1))
#    
#     Square LET
#    plotPolygon(make_square_let_gen_reg(cut_width=1, flexure_width=1 ,junction_length=3, edge_space=1.5, stem_width=1, num_flex=3, inside_start=False))
#    plotPolygon(make_square_let_unit(cut_width=1, flexure_width=1, junction_length=3, edge_space=1.5, stem_width=1, num_flex=3, inside_start=False))
#    
#     Triangular LET
#    plotPolygon(make_triangular_let_gen_reg(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=2,stem_width=1,num_flex=3,inside_start=False))
#    plotPolygon(make_triangular_let_unit(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=2,stem_width=1,num_flex=3,inside_start=False))
#    plotPolygon(make_triangular_let_flexure(cut_width=0.5,flexure_width=1,junction_length=3,edge_space=1,stem_width=1,num_flex=3,inside_start=False))
#    
#     Hexagonal LET
#    plotPolygon(make_hexagonal_let_gen_reg(cut_width=1, flexure_width=2, junction_length=6, edge_space=4, stem_width=2, num_flex=3, inside_start=False)) # 
#    plotPolygon(make_hexagonal_let_unit(cut_width=1, flexure_width=2, junction_length=6, edge_space=1, stem_width=1, num_flex=3, inside_start=False)) # BUG fail due to numerical distortions
#    plotPolygon(make_hexagonal_let_flexure(cut_width=0.5, flexure_width=1, junction_length=3, edge_space=1, stem_width=1, num_flex=3, inside_start=False)) # not reliable (due to scaling?)
#    
#    
#    # Switchbacks
#    # Comment: a little tweaking is necessary on the 'default'
#    plotPolygon(make_triangular_switchback_gen_reg(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=4, side_cut=1))
#    plotPolygon(make_hexagonal_switchback_gen_reg(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=4, side_cut=1))
#    plotPolygon(make_hexagonal_switchback_tile(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=5, side_cut='default'))
#    plotPolygon(make_rectangular_switchback_unit(num_turns=1, width_stem=1, length_flex=10, cut_width=1, width_flex=2))
#    plotPolygon(make_triangular_switchback_tile(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=2, side_cut=1))
#    plotPolygon(make_square_switchback_tile(cut_width=1, flexure_width=2, junction_length=5, edge_space=3, num_flex=2, side_cut=1))
#    plotPolygon(make_hexagonal_switchback_tile(cut_width=0.5, flexure_width=1, junction_length=2, edge_space=1, num_flex=3, side_cut=1))
#    
#    ## Coil
#    unit = (make_square_coil_unit(cut_width=1, flexure_width=1))
#    unitA = unit.buffer(0.01)
#    plotPolygon(make_square_coil_unit(cut_width=1, flexure_width=1))
#    
#    
#    #Surfaces
#    plotPolygon(map_p2_let(width_stem=1,length_flex=4,height_stem=0.2,width_flex=1,skew_angle=45,ncell_x=2, ncell_y=4))
#    plotPolygon(map_pmg_let(width_stem=1, length_flex=4, height_stem=0.2, width_flex=1, skew_angle=45, ncell_x=2, ncell_y=10))
#    
#    hex_unit = make_triangular_let_unit(cut_width=1,flexure_width=2,junction_length=4,edge_space=4,stem_width=1,num_flex=3,inside_start=False)
#    mapped_hex = map_surface(unit_cell=hex_unit, ncell_x=3, ncell_y=3, lattice='rhombic')
#    plotPolygon(mapped_hex)
#    
#    
#    #TEST CALL
#    polygon = make_square_let_unit(cut_width=1, flexure_width=1, junction_length=3, edge_space=1.5, stem_width=1, num_flex=3, inside_start=False)
#    polygon = make_square_let_unit(cut_width=1,
#                                   flexure_width=1,
#                                   junction_length=3,
#                                   edge_space=1.5,
#                                   stem_width=1,
#                                   num_flex=3,
#                                   inside_start=False)
#    print(polygon.type)
    
    

#
#"""
#shape/lattice (sq, rec, rhomb (tri, hex...), parallell)
#flex type (let, sb...)
#hierarc (gen_reg, unit)
#
#if let_sq -> give some paramters, make unit?
#- tiles not a focus...
#
#Unit(shape, flextype)
#Unit.set_paramter(**kwargs)
#- update polygon...
#
#Unit.present_paramters()
#
#which gives meaning to vary...
#thickness = const
#cut_width
#flexure_width - cut_width/flexure_width? (might be dependent)
#junction_length - independent - but want to be as small as possible?
#stem_width - dependent on flexure width? (stress in torsion vs bending- Ip vs Ib)
#num_flex
#inside_start (Only vary in one test. once...)
#side_cut - dependent on cut_width = assumption
#
#
#if 'word' in 'string':
#    do something
#
#default paramters = 1
#
#"""
