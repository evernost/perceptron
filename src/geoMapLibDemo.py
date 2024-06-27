# -*- coding: utf-8 -*-
# ========================================================================================
# Module name     : geoMapLibDemo
# File name       : geoMapLibDemo.py
# Purpose         : simple demo of the geoMapLib
# Author          : QuBi (nitrogenium@outlook.fr)
# Creation date   : 13/04/2020 (Covid19 lockdown time, yay!)
# ========================================================================================

# ========================================================================================
# Description
# ========================================================================================
# Basic library to experiment with perceptron!

# ========================================================================================
# Libraries declaration
# ========================================================================================
import geoMapLib as GML


mapGen = GML.MapGen()

nPts = 50000

mapGen.addRegion(GML.region(name = "tri_0",  testFunc = GML.triangleTest,  geometry = {"center": {"x": 0.1, "y": 0.2}, "r": 0.8, "angle": -30.0}, layer = 1))
mapGen.addRegion(GML.region(name = "ell_0",  testFunc = GML.ellipseTest,   geometry = {"center": {"x": 0.3, "y": 0.2}, "a": 0.5, "b": 0.3, "angle": 20.0}, layer = 2))
mapGen.addRegion(GML.region(name = "rect_0", testFunc = GML.rectangleTest, geometry = {"center": {"x": -0.5, "y": -0.4}, "a": 0.4, "b": 0.6, "angle": -10.0}, layer = 2))
mapGen.addRegion(GML.region(name = "rect_1", testFunc = GML.rectangleTest, geometry = {"center": {"x": 0.7, "y": -0.4}, "a": 1.5, "b": 0.3, "angle": 85.0}, layer = 3))
mapGen.addRegion(GML.region(name = "tri_1",  testFunc = GML.triangleTest,  geometry = {"center": {"x": 0.0, "y": -0.75}, "r": 0.2, "angle": -10.0}, layer = 4))




# Build the database
(pointsCoord, pointsClass, pointsRegionName) = mapGen.build(nPts)

mapGen.show() 

print()
