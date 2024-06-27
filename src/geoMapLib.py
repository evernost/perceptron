# -*- coding: utf-8 -*-
# =============================================================================
# Module name     : geoMapLib
# File name       : geoMapLib.py
# Purpose         : generation and sampling of 2D maps with geometric figures.
# Author          : QuBi (nitrogenium@outlook.fr)
# Creation date   : 14/04/2020 (Covid19 lockdown time, yay!)
# =============================================================================

# =============================================================================
# Description
# =============================================================================
# Define a custom 2D map with geometric figures (triangles, ellipses, etc.) and
# then generate a dataset by sampling the map.

# =============================================================================
# Libraries declaration
# =============================================================================
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.cm      as cm



# =============================================================================
# TODO
# =============================================================================
# > Add new geometric figures
# > Add boolean operations






def ellipseTest(inputPoint, geometry) :
  """
  Description:
  Test if a point belongs to an ellipse of a given geometry.
  Input point must be array([[x, y]]

  Known limitations:
  TODO

  Examples:
  TODO
  """  
  
  # Center the input
  X = inputPoint[0] - geometry["center"]["x"]
  Y = inputPoint[1] - geometry["center"]["y"]
  
  # Rotate and normalize
  t_x = 0
  t_x = t_x + X*np.cos(geometry["angle"]*2*np.pi/360.0)
  t_x = t_x + Y*np.sin(geometry["angle"]*2*np.pi/360.0)
  t_x = t_x/geometry["a"]

  t_y = 0
  t_y = t_y - X*np.sin(geometry["angle"]*2*np.pi/360.0)
  t_y = t_y + Y*np.cos(geometry["angle"]*2*np.pi/360.0)
  t_y = t_y/geometry["b"]

  return ((t_x**2.0) + (t_y**2.0) <= 1.0)





def rectangleTest(inputPoint, geometry) :
  """
  Description:
  Test if a point belongs to a rectangle of a given geometry.

  Known limitations:
  TODO

  Examples:
  TODO
  """  

  # Center the input
  X = inputPoint[0] - geometry["center"]["x"]
  Y = inputPoint[1] - geometry["center"]["y"]
  
  # Rotate and normalize
  t_x = 0
  t_x = t_x + X*np.cos(geometry["angle"]*2*np.pi/360.0)
  t_x = t_x + Y*np.sin(geometry["angle"]*2*np.pi/360.0)
  t_x = t_x/geometry["a"]

  t_y = 0
  t_y = t_y - X*np.sin(geometry["angle"]*2*np.pi/360.0)
  t_y = t_y + Y*np.cos(geometry["angle"]*2*np.pi/360.0)
  t_y = t_y/geometry["b"]

  return ((np.abs(t_x) <= 0.5) and (np.abs(t_y) <= 0.5))




def triangleTest(inputPoint, geometry) :
  """
  Description:
  Test if a point belongs to a triangle of a given geometry.

  Known limitations:
  TODO

  Examples:
  TODO
  """  

  # Center the input
  X = inputPoint[0] - geometry["center"]["x"]
  Y = inputPoint[1] - geometry["center"]["y"]
  
  # Rotate and normalize
  t_x = 0
  t_x = t_x + X*np.cos(geometry["angle"]*2*np.pi/360.0)
  t_x = t_x + Y*np.sin(geometry["angle"]*2*np.pi/360.0)

  t_y = 0
  t_y = t_y - X*np.sin(geometry["angle"]*2*np.pi/360.0)
  t_y = t_y + Y*np.cos(geometry["angle"]*2*np.pi/360.0)

  c1 = (t_y >= -geometry["r"]/2.0)
  c2 = (t_y <= geometry["r"] + np.sqrt(3)*t_x)
  c3 = (t_y <= geometry["r"] - np.sqrt(3)*t_x)

  return (c1 and c2 and c3)




class region :

  def __init__(self, name, testFunc, geometry, layer) :
    self.name       = name
    self.testFunc  = testFunc
    self.geometry   = geometry
    
    if (layer >= 0) :
      self.layer = layer
    else :
      raise Exception("[geoMapLib] Negative layers ID are reserved.")


  
  def isMember(self, point) :
    return self.testFunc(point, self.geometry)






# -----------------------------------------------------------------------------
# Map Generator class
# -----------------------------------------------------------------------------
class MapGen : 
  
  # ---------------------------------------------------------------------------
  # Initialisation 
  # ---------------------------------------------------------------------------
  def __init__(self, xRange = [-1.0, 1.0], yRange = [-1.0, 1.0]) :
    self.regions  = []
    self.nRegions = 0
    self.xMin = xRange[0]
    self.xMax = xRange[1]
    self.yMin = yRange[0]
    self.yMax = yRange[1]
    self.nPts = 0



  # ---------------------------------------------------------------------------
  # ADD_REGION method
  # ---------------------------------------------------------------------------
  def addRegion(self, region) :
    """
    Declare a new region to the list of regions.
    Region has: 
    - a name
    - 
    
    Example:
    > XXX.add_region(name = "ell0",   test_func = ellipseTest,   geometry = {"center": {"x": x0, "y": y0}, "a": 0.5, "b": 0.1, "angle" = 45.0})
    > XXX.add_region(name = "ell0",   test_func = ellipseTest,   geometry = {"center": {"x": x0, "y": y0}, "a": 0.5, "b": 0.1, "angle" = 45.0})
    > XXX.add_region(name = "rect0",  test_func = rectangleTest, geometry = {"center": {"x": x0, "y": y0}, "a": 0.5, "b": 0.1, "angle" = 45.0})
    """
    
    self.regions.append(region)
    self.nRegions += 1



  # ---------------------------------------------------------------------------
  # BUILD method
  # ---------------------------------------------------------------------------
  def build(self, nPts) :
    """
    Build an output dictionnary that as the same keys as dictionnary
    "self.regions", and initialize each key with an empty list.
    Also, add a null region ("none").
    
    """


  
    # Sample the 2D plane
    points_x = np.random.uniform(low = self.xMin, high = self.xMax, size = (nPts, 1))
    points_y = np.random.uniform(low = self.yMin, high = self.yMax, size = (nPts, 1))
    self.pointsCoord = np.hstack((points_x, points_y))
    self.pointsClass = np.ones((nPts, 1))*(-1)
    self.pointsRegionName = ["" for _ in range(nPts)]
    self.nPts = nPts

    # Classify points
    for i in range(nPts) :
      for currRegion in self.regions :
        if (currRegion.isMember(self.pointsCoord[i, :])) :
          if (currRegion.layer > self.pointsClass[i, 0]) :
            self.pointsClass[i, 0]   = currRegion.layer
            self.pointsRegionName[i] = currRegion.name
    
    return (self.pointsCoord, self.pointsClass, self.pointsRegionName)



  # --------------------------------------------------------------------------------------
  # SHOW method
  # --------------------------------------------------------------------------------------
  def show(self) :
    
    # colors = np.empty((self.nPts, 4))
    # colorMap = cm.rainbow(np.linspace(0, 1, self.nRegions))

    # for (i, currClass) in enumerate(self.pointsClass) :
    #   if (currClass[0] < 0) :
    #     colors[i,:] = [0, 0, 0, 1.0]

    #   else :
    #     colors[i,:] = colorMap[int(currClass[0])-1,:]

    plt.figure()
    plt.scatter(self.pointsCoord[:,0], self.pointsCoord[:,1],  c = self.pointsClass, cmap = 'viridis', s = 0.1, marker = "o")
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Generated database")
    #plt.show(block = False)







