# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:19:54 2020

@author: rubenandrebarreiro
"""

# Import NumPy Python's Library as np
import numpy as np

# Import PyPlot Sub-Module,
# from Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# Import cluster.KMeans Sub-Module,
# from SciKit-Learn Python's Library as k_means
from sklearn.cluster import KMeans as k_means

# Import io.imsave and io.imread Sub-Modules,
# from SciKit-Learn's Image Processing Python's Library as im_save and im_read
from skimage.io import imsave as im_save, imread as im_read


# The file of the Vegetables' Image
vegetables_img_file = "../files/imgs/vegetables.png"


# Read the file of the Vegetables' Image
vegetables_img = im_read(vegetables_img_file)

# The Shape of the file of the Vegetables' Image
# (i.e., Width, Height and Depth)
width, height, depth = vegetables_img.shape

# Reshape the file of the Vegetables' Image as Columns
columns = np.reshape( vegetables_img/255.0, (width * height, depth) )


def representation_3d_for_color_space(columns):
    
    # Initialise the Figure, for plotting
    fig = plt.figure(figsize=(8,8))
    
    # Set the Projection as 3D
    ax = fig.add_subplot(111, projection='3d')
    
    # Set the Title of the Plot
    ax.title.set_text('3D Representation for the (R,G,B) Colors\' Space')
    
    # Set a Scatter Plot, for the columns
    ax.scatter(columns[:,0], columns[:,1], columns[:,2], c=columns, s=10)
    
    
    # Set the Labels for the axis (X,Y,Z) and
    # the associations between them and the colors (R,G,B)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    
    # The vectors for the limits of the axis
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)
    
    plt.savefig('imgs/vegetables-3d-representation-color-rgb-space.png',
                dpi=200, bbox_inches='tight')
    
    plt.show()
    
    plt.close()
    

def k_means_colors(columns, num_clusters_centroids_colors = 64):
    
    image_array = columns[:,:3]
    
    k_means_clustering = k_means(n_clusters = num_clusters_centroids_colors).fit(image_array)

    labels = k_means_clustering.predict(image_array)

    centroids = k_means_clustering.cluster_centers_
    
    
    # Set the Title of the Plot
    plt.title('KMeans for the Vegetables Image,\nwith {} Centroids/Clusters'.format(num_clusters_centroids_colors))
    
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x',
                color='k',s=200, linewidths=5)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x',
                color='w',s=150, linewidths=2)
    
    plt.savefig('imgs/vegetables-k-means-plot-' + str(num_clusters_centroids_colors) + '-centroids.png',
                dpi=200, bbox_inches='tight')

    plt.show()
    
    plt.close()
    
        
    centroids_columns = np.zeros(columns.shape)
    
    for ix in range(columns.shape[0]):
        
        centroids_columns[ix,:] = centroids[labels[ix]]
    
    
    final_img = np.reshape( centroids_columns, (width, height, 3) )
    
    
    im_save('imgs/vegetables-quality-for-{}-colors.png'.format(num_clusters_centroids_colors), final_img)
    



representation_3d_for_color_space(columns)
    
k_means_colors(columns, 64)
k_means_colors(columns, 8)