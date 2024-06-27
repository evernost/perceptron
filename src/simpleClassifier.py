# -*- coding: utf-8 -*-
# ========================================================================================
# Module name     : simpleClassifier
# File name       : simpleClassifier.py
# Purpose         : simple classifier example with the nnLib
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
# Official imports
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.cm      as cm
import time

# User imports
import nnLib
import geoMapLib

reg_to_vect = {
  "ell_0" : np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T,
  "rect_0": np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]).T,
  "rect_1": np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).T,
  "rect_2": np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]).T,
  "tri_0" : np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).T,
  "none"  : np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T
}

def learning_rate(progression, mu_min = 0.0001, mu_max = 0.007) :
  prog_thresh = 0.01
  if (progression < prog_thresh) :
    return mu_max
  else :
    return ((prog_thresh*(mu_max - mu_min)/progression) + mu_min)

def confusion(input_vector) :
  norm_1 = np.linalg.norm(input_vector, ord = 1)
  norm_2 = np.linalg.norm(input_vector, ord = 2)
  
  if (norm_2 == 0.0) :
    return 0.0
  else :
    return ((norm_1/norm_2)-1)/(np.sqrt(input_vector.shape[0])-1)

def confusion_to_index(confusion, N_bins) :
  if (confusion >= 1.0) :
    return N_bins-1
  else :
    return int(np.round(confusion*N_bins))




# ========================================================================================
# Main code
# ========================================================================================
# Note: run code in Python terminal
# > exec(open('hello.py').read())
# Set breakpoint
# > pdb.set_trace()

N_database = 500

regions = []
regions.append(geoMapLib.region(name = "tri_0",   test_func = geoMapLib.s_triangle,  geometry = {"center": {"x": 0.1, "y": 0.2}, "r": 0.8, "angle": -30.0}, layer = 1))
regions.append(geoMapLib.region(name = "ell_0",   test_func = geoMapLib.s_ellipse,   geometry = {"center": {"x": 0.3, "y": 0.2}, "a": 0.5, "b": 0.3, "angle": 20.0}, layer = 2))
regions.append(geoMapLib.region(name = "rect_0",  test_func = geoMapLib.s_rectangle, geometry = {"center": {"x": -0.5, "y": -0.4}, "a": 0.4, "b": 0.6, "angle": -10.0}, layer = 3))
regions.append(geoMapLib.region(name = "rect_1",  test_func = geoMapLib.s_rectangle, geometry = {"center": {"x": -0.5, "y": 0.6}, "a": 0.3, "b": 0.3, "angle": 10.0}, layer = 4))
regions.append(geoMapLib.region(name = "rect_2",  test_func = geoMapLib.s_rectangle, geometry = {"center": {"x": 0.2, "y": -0.5}, "a": 0.5, "b": 0.5, "angle": 45.0}, layer = 4))

# Setup the generator
db_gen = geoMapLib.db_generator(regions)

# Build the database
database = db_gen.build(N_database)

# ----------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------
mu      = 5e-5
mu_min  = 5e-7
mu_max  = 9e-4

innov_rate = 0.0

layers_dim  = [2,5,5,5,6]
N_epoch     = 200
N_subset    = 200

# ----------------------------------------------------------------------------------------
# Build the network
# ----------------------------------------------------------------------------------------
network_0 = nnLib.network(layers_dim,mu)

# ----------------------------------------------------------------------------------------
# Learning
# ----------------------------------------------------------------------------------------
e_mat = np.zeros((N_epoch,1))
s_mat = np.zeros((N_epoch,1))

# Debugging tools
v_out = np.zeros((N_epoch,6))

progress      = 0.0
progress_lap  = -1.0

N_bins = 4

# network_0.restore("test_1.txt")
fig_err = plt.figure()
e_array = np.zeros((N_database,N_epoch))

# Present the whole database several times to the network
for epoch in range(N_epoch) :
  print("*************** Epoch " + str(epoch) + "/" + str(N_epoch) + " ***************")
  
  # Start stopwatch
  t0 = time.time()

  # Initialise the mean error of this epoch
  e_epoch = 0.0
  s_epoch = 0.0

  e_worst       = 0.0
  e_worst_index = 0
  e_worst_vect  = 0.0


  # Loop on the points of the database
  for point_index in range(N_database) :
    
    # Create point vector
    curr_point = database[point_index % N_subset]["coord"]

    # -------------------
    # Forward propagation
    # -------------------
    infer = network_0.forwards(curr_point)

    # Build the average entropy
    s_epoch = s_epoch + confusion(infer)

    # Evaluate the error vector
    e_vect = infer - reg_to_vect[database[point_index % N_subset]["region"]]
    e_norm = float(e_vect.T @ e_vect)
    e_array[point_index,epoch] = e_norm
    # e_array[point_index,epoch] = confusion(infer)

    # Add it to the global error for mean estimation
    e_epoch = e_epoch + e_norm

    # DEBUG: follow one random output
    if (point_index == 153) :
      # v_out[[epoch],:] = infer.T
      v_out[[epoch],:] = e_vect.T

    # Record the point with worse prediction in this epoch
    if (e_norm > e_worst) :
      e_worst       = e_norm
      e_worst_index = point_index
      e_worst_vect  = e_vect

    # --------------------
    # Backward propagation
    # --------------------
    network_0.backwards(e_vect)

    # ------------------
    # Update the network
    # ------------------
    network_0.update()

  # Hit harder on the 50 points with highest error >:(
  e_array_idx_sort = np.argsort(-e_array[:,epoch], axis = 0)
  
  for i in range(100) :
    for j in range(10) :
      curr_point  = database[e_array_idx_sort[j]]["coord"]
      curr_reg    = database[e_array_idx_sort[j]]["region"]
      
      infer = network_0.forwards(curr_point)
      e_vect = infer - reg_to_vect[curr_reg]
      network_0.backwards(e_vect)
      network_0.update()

  # End of the epoch. 
  # Collect and print current results
  e_mat[epoch] = e_epoch/N_database
  s_mat[epoch] = s_epoch/N_database

  print("Epoch stats:")
  print("> mean error  : {:.4f} (worst: {:.4f}, index = {:d})".format(e_epoch/N_database,e_worst,e_worst_index))
  print("> mean entropy: {:.4f}".format(s_epoch/N_database))
  print(e_worst_vect)
  print("")

  # Propose a new learning rate considering the current error progression.
  if (epoch > 0) :
    print("Re-evaluating learning rate...")
    mu = learning_rate(float(np.abs(e_mat[epoch] - e_mat[epoch-1])), mu_min, mu_max)
    # print("> current mu = {0:.5f}".format(mu))
    # print("> new mu     = {0:.5f} ({1:.1f}% of mu_max)".format(mu, 100.0*mu/mu_max))
    print("> mu = {0:.5f}".format(mu))
    
  #   # The more epoch we have gone through, the lower gets the highest learning rate reachable.
  #   # mu_max = (1-innov_rate)*mu_max + innov_rate*mu_min
  #   print("")
    
    network_0.set_learning_rate(mu)

  # N_x = 150
  # N_y = 150
  # classif = np.zeros((N_y,N_x))

  # for (id_x, x) in enumerate(np.linspace(-1.2, 1.2, N_x)) :
  #   for (id_y, y) in enumerate(np.linspace(-1.2, 1.2, N_y)) :
  #     v = network_0.forwards(np.array([[x,y]]).T)
  #     classif[id_y, id_x] = np.argmax(v)

  # plt.imshow(classif, cmap = cm.rainbow, origin='lower')
  # plt.gca().set_aspect('equal', adjustable='box')
  # plt.title("Classification map (entropy = {0:.5f})".format(s_epoch/N_database))
  # plt.show(block = False)
  # plt.pause(0.01)

  if (epoch > 5) :
    fig_err.clear()
    plt.grid(True)
    plt.plot(np.arange(N_database),np.sort(e_array[:,(epoch-5):epoch],axis = 0))
    plt.show(block = False)
    plt.pause(0.01)

  # fig_err.clear()
  # plt.grid(True)
  # plt.plot(np.arange(N_database),e_bin)
  # plt.show(block = False)
  # plt.pause(0.01)

  # Print elapsed time
  t1 = time.time()
  print("Epoch time: {0:.1f}s".format(t1-t0)) 
  print("")


plt.figure()
plt.plot(np.arange(N_epoch),v_out)
plt.show()

# Save the network
# network_0.save("test_1.txt")

# ----------------------------------------------------------------------------------------
# Plot the classification result
# ----------------------------------------------------------------------------------------
N_x = 50
N_y = 50
classif = np.zeros((N_y,N_x))
conf    = np.zeros((N_y,N_x))

network_0.profile_enable()

for (id_x, x) in enumerate(np.linspace(-1.2, 1.2, N_x)) :
  for (id_y, y) in enumerate(np.linspace(-1.2, 1.2, N_y)) :
    v = network_0.forwards(np.array([[x,y]]).T)
    classif[id_y, id_x] = np.argmax(v)
    conf[id_y, id_x]    = int(256*confusion(v))

plt.figure()
plt.imshow(classif, cmap = cm.rainbow, origin = 'lower')
plt.gca().set_aspect('equal', adjustable = 'box')
plt.title("Classification results")
plt.show(block = False)

plt.figure()
plt.imshow(conf, cmap = cm.rainbow, origin = 'lower')
plt.gca().set_aspect('equal', adjustable = 'box')
plt.title("Confusion plot")
plt.show(block = False)

network_0.profile_show()

# ----------------------------------------------------------------------------------------
# Plot the actual region map
# ----------------------------------------------------------------------------------------
db_gen.show(database)

# ----------------------------------------------------------------------------------------
# Last operations
# ----------------------------------------------------------------------------------------
print("")
print("What is next?")
print("- quit.........(q)")
print("- explore......(e)")
print("")
u_input = input("> ")

if (u_input == "q") :
  quit()
elif (u_input == "e") :
  print("Eval mode (enter 'q' to exit)")
  while True :
    x = input("p.x = ")
    y = input("p.y = ")

    if ((x == "q") or (y == "q")) :
      quit()
    else :
      classif = 100.0*network_0.forwards([float(x),float(y)])

      print(classif)
      print()




