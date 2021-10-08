#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 04 10:04:04 2021

@author: Florian Rückerl
"""

# =============================================================================
# Script to analyze IMARIS data from two-photon imaging (PSD95-venus and Alexa594 cell body)
# for the article:
#
# *Developmental emergence of two-stage nonlinear synaptic integration in cerebellar interneurons*
# authors: Célia Biane, Florian Rückerl, Therese Abrahamsson, Cécile Saint-Cloment,
# Jean Mariani , Ryuichi Shigemoto, David A. DiGregorio, Rachel M. Sherrard, and
# Laurence Cathala
#
# This is an outline for analyzing the data, it will need to be adapted
# for individual experiments. Data is saved as tables in excel format and/or as
# .pckl files
# =============================================================================
# Outline:
# Step 1: Dendritic structure and associated spots
#   a) create dendritic structure as a Tree (networkx) from IMARIS .ims file
#   b) select spots detected by IMARIS by intensity and size
#   c) find PSD95 spots associated with dendrites (mindisty <200nm)) and
#      calculate distances along dendrite for each spot
#   d) calculate distance for dendrite segments
# Step 2: Soma and associated spots
#   a) Determine spots that are associated with the soma (requires loading the
#       actual image data from .ims file via ndimage)and thresholds determined in Fiji
# Step 3: Analysis of spot and denderite data
#   a) Create histograms of spot and dendrite distances (including and excluding the soma)
#   b) Count branch points on primary dendrites
# =============================================================================
#
# Programming environment:
# Python 3.7.6
# conda 4.10.3
# spyder 4.0.1
#
# Used module versions:
# h5py        2.10.0
# networkx    2.5.1
# pandas      1.2.4
# scipy       1.6.2
# matplotlib  3.3.4
# numpy       1.20.2
# =============================================================================

# =============================================================================
### Import section
# =============================================================================

# module for reading IMARIS files
import h5py

# data tree manipulation
import networkx as nx

#  modules for data format
import pandas as pd
import pickle

# image maniopulation
from scipy import ndimage

# modules for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# general modules
import sys
import numpy as np
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime as dt

# =============================================================================
### Custom functions section
# =============================================================================


# =============================================================================
### main dendritic tree analysis function
# =============================================================================



def Analyze_Dendritic_Tree(Input, mainfolder, plot_flag = False):
    """
    Creates dendritic tree from IMARIS file for further analysis.

    The sections to generate the dendritic tree and calculate the length along the branches
    was initially created by Dmitry Ershov (Image Analysis Hub, Institut Pasteur)

    Parameters
    ----------
    Input : list of tupel [(,)]
        list of tuple containing the filename of the IMARIS file and the ID of
        the root node (determined in IMARIS)
    mainfolder : pathlib Path
        path to folder containing the IMARIS files or subfolders
    plot_flag : boolean
        flag for plotting the skeletal structure of the dendritic tree; Default is False
    Returns
    -------
    Data : list of dictionaries
        dictionary entries:
                {'file':currentFile,
                'spots'     : coordinates of PSD95 spots from IMARIS
                'vertices'  : coordinates of dendrite vortices from IMARIS
                'diameters' : estimated dendrite diameters from IMARIS
                'segment_length'    : calculated length of dendrite segments between branching points
                'Tree'      : Tree structure of dendrites
                'leaves'    : end points of dendritic branches
                'root'      : root of the dendritic tree (NOT the start of primary dendrites)
                'dendrite_length'   : length of dendrites }

    """
    #structure to store data
    Data = []
    for plotind, In in enumerate(Input):
        currentFile = In[0]
        root = In[1]
        D = {'file':currentFile,
                'spots':[], # coordinates od PSD95 spots from IMARIS
                'vertices':[], # coordinates of dendrite vortices from IMARIS
                'diameters':[], # estimated dendrite diameters from IMARIS
                'segment_length': [],#calculated length of dendrite segments between branching points
                'Tree':[],# Tree structure of dendrites
                'leaves':[],# end points of dendritic branches
                'root':[], # root of the dendritic tree (NOT the start of primary dendrites)
                'dendrite_length':[]} # length of dendrites }

        # full path of current file
        f0 = mainfolder / currentFile

        ## 1. read data from hdf5 file
        start = timer()
        print('Reading the data of file ',currentFile,'...', end = '')
        with h5py.File( f0, 'r') as f:
             # list of segments as a pair of vertices.
            fils_graph_segms        = f['Scene']['Content']['Filaments0']['Graphs']['Segments'][()]
            # coords of vertices
            fils_graph_verts        = f['Scene']['Content']['Filaments0']['Graphs']['Vertices'][()][:, 0:3]
            # diameter of each vertice (estimated inside IMARIS)
            fils_graph_diams        = f['Scene']['Content']['Filaments0']['Graphs']['Vertices'][()][:, 3]
            # root index of the tree (from IMARIS file, might be corrected below)
            root_start = fils_graph_verts[root]
        #  store in dictionary
        D['vertices'] = fils_graph_verts
        D['diameters'] = fils_graph_diams
        print('done.')#

        ## 2. build the graph of the dendritic tree for analysis
        print('Building the graph ...', end = '')
        gr = nx.Graph()
        # go over each segment, remove isolates (non connected segments), and build graph
        for seg in fils_graph_segms:
            v1, v2              = seg
            p1, p2              = fils_graph_verts[v1], fils_graph_verts[v2]

            seg_len             = np.linalg.norm( p1 - p2 )
            gr.add_edge( seg[0], seg[1], weight=seg_len )
        gr.remove_nodes_from(list(nx.isolates(gr)))

        print('done.')

        ## 3. rebuit graph as a Tree
        print('rebuilding graph as tree...', end = '')
        # check if root is vertice in vertices set (this is rarely necessary)
        if np.sum(root_start[:]) != 0:
            d               = fils_graph_verts - root_start
            # choose root coordinate in vertice set closest to initial vertice
            root       = np.linalg.norm(d, axis=1).argmin()
        else:
            maxnode = max(dict(gr.degree()).items(), key = lambda x : x[1])
            root = maxnode[0]
        print('using root [',root,']  = '+str(fils_graph_verts[root])+'...',end = '')
        # rebuilt tree
        T = nx.bfs_tree(gr, root)


        ## 4. find end points of tree (Leaves)
        leaves = [x for x in T.nodes() if T.out_degree(x)==0 and T.in_degree(x)==1]


        # 5. plot dendritic Tree
        # loop through leaves for plotting
        if plot_flag:
            plt.figure()
            for l in leaves:
                dend_list = nx.shortest_path( T, source=root, target=l )
                d = pd.DataFrame(fils_graph_verts[dend_list])#-

                d.columns = ['x','y','z']
                plt.plot(d['x'],d['y'],color=colors['black'],alpha=0.25)
                plt.title = f0.stem
            r = fils_graph_verts[root]
            plt.plot(r[0], r[1],'m+')

        # store in dictionary
        D['Tree'] = T
        D['leaves'] = leaves
        D['root'] = root
        print('done.')

        # 6. get segment statistics (not strictly necessary)
        print('Getting segment statistics...', end = '')

        for seg in T.edges:
            # get start and end point
            p1, p2              = fils_graph_verts[seg[0]], fils_graph_verts[seg[1]]
            # calclulate segment length
            seg_len             = np.linalg.norm( p1 - p2 )
            # store in dictionary
            D['segment_length'].append(seg_len)

        print('done.')

        # 7. calculate dendrite length for cell
        print('calculating dendrite length...', end = '')
        dend = []
        for vertind in range(len(fils_graph_verts)):
            # ensure that node exists
            if T.has_node(vertind):
                # get list of nodes/vertices from root to vertice
                dend_list   = nx.shortest_path( T, source=root, target=vertind )
                # get xx,y,z coordinates of  vertex list
                d           = fils_graph_verts[dend_list]
                dendlen     = 0
                seg_len     = []
                #calculate length along path
                for i in range(len(dend_list)-1):
                    v1, v2  = dend_list[i], dend_list[i+1]
                    seg_len = np.linalg.norm( fils_graph_verts[v1] - fils_graph_verts[v2] )
                    dendlen += seg_len
                dend.append(dendlen)

        # store in dictionary
        D['dendrite length'] = dend
        print('done in '+'{:.1f}'.format(timer()-start)+' seconds\n---\n')
        Data.append(D)

    return Data

# =============================================================================
### image and IMARIS analysis functions
# =============================================================================
def get_Data_Attribute_as_string(f,Attribute):
    """
    Read attribute from IMARIS file as string

    Parameters
    ----------
    f : h5py file
        HDF5 type of file opened using h5py
    Attribute : string
        keyword for the attribute

    Returns
    -------
    Attr : string
        attribute associarted with keyword specified in parameter Attribute

    """
    AttrList = list(f.attrs[Attribute])
    Attr = ''
    for A in AttrList:
        Attr = Attr+ A.decode('utf-8')
    return Attr

#
def get_Data_Attribute_as_float(f,Attribute):
    """
    Read attribute from IMARIS file as string and convert to float

    Parameters
    ----------
    f : h5py file
        HDF5 type of file opened using h5py
    Attribute : string
        keyword for the attribute

    Returns
    -------
    Attr : float
        attribute associated with specified keyword ($Attribute)

    """
    Attr = float(get_Data_Attribute_as_string(f,Attribute))
    return Attr

# read dimension of image from IMARIS file
def get_Data_Dimensions(f):
    """
    Retrieves size of image stack from IMARIS file

    Parameters
    ----------
    f : h5py file
        HDF5 type of file opened using h5py.

    Returns
    -------
    dictionary
        dictionary containing the offset and width in X,YZ and the unit as a string.

    """
    df = f['DataSetInfo']['Image']
    dim = {'X': [0,0],
           'Y': [0,0],
           'Z': [0,0],
           'unit' :''}
    minX = get_Data_Attribute_as_float(df,'ExtMin0')
    minY = get_Data_Attribute_as_float(df,'ExtMin1')
    minZ = get_Data_Attribute_as_float(df,'ExtMin2')

    dim['X'] = [minX, get_Data_Attribute_as_float(df,'ExtMax0') - minX]
    dim['Y'] = [minY, get_Data_Attribute_as_float(df,'ExtMax1') -minY]
    dim['Z'] = [minZ, get_Data_Attribute_as_float(df,'ExtMax2') - minZ]
    dim['unit'] = get_Data_Attribute_as_string(df,'Unit')
    return pd.DataFrame(dim, index = ['offset', 'width'])


def get_Data_Dimensions_in_pixel(f):
    """
    Return dimension of images stack from IMARIS file in pixel

    Parameters
    ----------
    f : h5py file
        HDF5 type of file opened using h5py.

    Returns
    -------
    DataFrame (pandas)
        DataFram containing the min and max image pixel coordinates in X,Y,Z. Unit is px.

    """
    df = f['DataSetInfo']['Image']
    dim = {'X': [get_Data_Attribute_as_float(df,'X')],
           'Y': [get_Data_Attribute_as_float(df,'Y')],
           'Z': [get_Data_Attribute_as_float(df,'Z')],
           'unit': 'px'}

    return pd.DataFrame(dim)

def add_pixel_position_from_coordinates_to_DF(DF, pixeldim, micronDF):
    """
    Calculate pixel positon in image to each coordinate of the detected spots
    IMARIS coordinates are in micrometer

    Parameters
    ----------
    DF : DataFrame (pandas)
        DataFrame containing at least the columns ['Position X','Position Y','Position Z'].
    pixeldim :  DataFrame (pandas)
        Contains image dimension (X,YZ) in pixel.
    micronDF :  DataFrame (pandas)
        Contains image dimension (X,YZ) in micrometer

    Returns
    -------
    DF :  DataFrame (pandas)
       DataFrame with added columns containing the pixel position corresponding to actual coordinates.

    """
    offset          = np.array(micronDF.loc['offset'][['X','Y','Z']])
    micron_dim      = np.array(micronDF.loc['width'][['X','Y','Z']])
    pixel_dim       = np.array(pixeldim.iloc[0][['X','Y','Z']])
    fact            = pixel_dim/micron_dim # pixel per micrometer
    position        = DF[['Position X','Position Y','Position Z']]
    pixel_position  = np.array((position-offset)*fact, dtype = 'int')
    pixDF           = pd.DataFrame(pixel_position,columns =['X','Y','Z'])
    DF = pd.concat([DF,pixDF], axis = 1)

    return DF

#
def get_pixel_position_from_coordinates(position,pixeldim,micronDF):
    """
    Get pixel position in image stack from XYZ  (micrometer) coordinates

    Parameters
    ----------
    position : array of float
        [X,Y,Z] coordinates in micrometer
    pixeldim :  DataFrame (pandas)
        Contains image dimension (X,YZ) in pixel.
    micronDF :  DataFrame (pandas)
        Contains image dimension (X,YZ) in micrometer

    Returns
    -------
    pixel_position : int
        pixel coordinates in images stack.

    """
    offset          = np.array(micronDF.loc['offset'][['X','Y','Z']])
    micron_dim      = np.array(micronDF.loc['width'][['X','Y','Z']])
    pixel_dim       = np.array(pixeldim.iloc[0][['X','Y','Z']])
    fact            = pixel_dim/micron_dim
    pixel_position  = np.array((position-offset)*fact, dtype = 'int')

    return pixel_position

#
def get_subpixel_position_from_coordinates(position,pixeldim,micronDF):
    """
    Get pixel position in image stack  (subpixel accuracy) from XYZ  (micrometer) coordinates

    Parameters
    ----------
    position : array of float
        [X,Y,Z] coordinates in micrometer
    pixeldim :  DataFrame (pandas)
        Contains image dimension (X,YZ) in pixel.
    micronDF :  DataFrame (pandas)
        Contains image dimension (X,YZ) in micrometer

    Returns
    -------
    subpixel_position : TYPE
        subpixel coordinates in images stack.

    """
    offset          = np.array(micronDF.loc['offset'][['X','Y','Z']])
    micron_dim      = np.array(micronDF.loc['width'][['X','Y','Z']])
    pixel_dim       = np.array(pixeldim.iloc[0][['X','Y','Z']])
    fact            = pixel_dim/micron_dim
    subpixel_position  = np.array((position-offset)*fact)

    return subpixel_position

#
def get_Zprofile(Stack, coord):
    """
    Get zprofile of image intensity at specified XY pixel coordinate

    Parameters
    ----------
    Stack : numpy array
        numpy array [X,Y,Z] containing the image intensity values.
    coord : [int,int]
        pixel coordiantes [X,Y].

    Returns
    -------
    numpy array
        array containing the intensity values of the image stack along Z

    """
    x   = coord[0]
    y   = coord[1]
    return Stack[:,y,x]


#

# =============================================================================
### spot and dendrite structure analysis
# =============================================================================

def get_dendrite_spot_distances(collectedFilteredData, Tree, mindist = 0.2):
    """
    Calculates the distance of all PSD95 spots to the dendrite

    Parameters
    ----------
    collectedFilteredData : DataFrame (pandas)
        DataFrame containing the.
    Tree : networkx tree
        networkx tree containing the dendrite structure.
    mindist : float
        minimal distance of spot to dendrite in mlicrometer (spots further away are rejected).
        Default is 0.2μm (200nm)

    Returns
    -------
    all_distances : DataFrame (pandas)
        Contains the Distance of PSD95 spots to the nearest Dendrite coordinate and additional information on the spot ([X,Y,Z] coordinates , intensity, diameter), as well as the dendrite (ID, diameter coordinate (micrometer))

    """
    all_distances = []
    for Spots in collectedFilteredData:
        spot_dendrite_distance = []
        filestem = Spots['file']
        print(filestem)
        #indices should be the same
        verts = [f['vertices'] for f in Tree if f['file'].split('_new')[0] == filestem][0]
        diameters = [f['diameters'] for f in Tree if f['file'].split('_new')[0] == filestem][0]
        check = 0
        for ind, v in enumerate(verts):
            d                   = v - Spots['spots'][['Position X','Position Y','Position Z']]
            distsDF             = pd.DataFrame(np.linalg.norm(d, axis=1), columns ={'dists'})
            distsDF['dists']    = distsDF['dists'] - diameters[ind]/2
            if mindist:
                close_dists         = distsDF.loc[distsDF['dists'] <=mindist]
            else:
                close_dists         = distsDF

            for index, row in close_dists.iterrows():
                spot_dendrite_distance.append({
                                      'spot-dendrite distance': row['dists'],
                                      'spot ID': Spots['spots'].iloc[index]['ID'],
                                      'spot intensity': Spots['spots'].iloc[index]['Intensity Center'],
                                      'spot diameter XY':  Spots['spots'].iloc[index]['Diameter X'],
                                      'spot diameter Z':  Spots['spots'].iloc[index]['Diameter Z'],
                                      'spot X': Spots['spots'].iloc[index]['Position X'],
                                      'spot Y': Spots['spots'].iloc[index]['Position Y'],
                                      'spot Z': Spots['spots'].iloc[index]['Position Z'],
                                      'dendrite ID': ind,
                                      'dendrite diameter': diameters[ind],
                                      'dendrite X': v[0],
                                      'dendrite Y': v[1],
                                      'dendrite Z': v[2]
                                      })
        spot_dendrite_DF = pd.DataFrame(spot_dendrite_distance)


        leaves = [f['leaves'] for f in Tree if f['file'].split('_new')[0] == filestem][0]
        root = [f['root'] for f in Tree if f['file'].split('_new')[0] == filestem][0]
        T= [f['Tree'] for f in Tree if f['file'].split('_new')[0] == filestem][0]
        for l in leaves:
            dend_list = nx.shortest_path( T, source=root, target=l )
            d = pd.DataFrame(verts[dend_list])#-
            d.columns = ['x','y','z']
        r = verts[root]
        all_distances.append({
                              'file': filestem,
                              'spots': spot_dendrite_DF
                              })
    return all_distances


#
def get_Spot_Soma_Distances(allDistances, collectedSoma, collectedTree,sizefilter):
    """
    Calculates distance of spot to soma (=start of primary dendrite branch)

    Parameters
    ----------
    allDistances : DataFrame
        created by get_dendrite_spot_distances(); contains information on the spots and the associated dendrite nodes
    collectedSoma : DataFrame (pandas)
        DESCRIPTION.
    collectedTree : list
        list of networkx trees; filename of dataset used to create tree is used as identifier.
    sizefilter : Boolean
        allows to reject spots sammler than (0.3x0.3x1.1 micrometer) .

    Returns
    -------
    allSoma : DataFrame (pandas)
        DataFrame containing the SOMA ID (netwokx node), Dendrite ID (networkx node), distance soma to dendrite (micrometer, and distance soma to root (micrometer).

    """
    allSoma = []
    for data in allDistances:
        if sizefilter:
            filtereddata = data['spots'][(data['spots']['spot diameter XY']> 0.3) & (data['spots']['spot diameter Z'] > 1.1)]
        else:
            filtereddata = data['spots']
        #data = allYoungDistances[0]
        filestem = data['file']
        print(filestem)
        T = [f['Tree'] for f in collectedTree if f['file'].split('_new')[0] == filestem][0]
        vertices = [f['vertices'] for f in collectedTree if f['file'].split('_new')[0] == filestem][0]
        root = [f['root'] for f in collectedTree if f['file'].split('_new')[0] == filestem][0]
        Soma = [f['soma data'] for f in collectedSoma if f['file'].stem.split('_new')[0] == filestem][0]
        Soma = Soma[['soma ID','dendrite ID','soma root distance']]
        soma_dist = []
        for ID in filtereddata['dendrite ID']:
           # spotID = [f for f in data['spots']['spot ID']]
            dend_list = []
            if T.has_node(ID):
                dend_list   = nx.shortest_path( T , source=root, target=ID )
                s = list_intersect(dend_list,Soma['dendrite ID'])
                if len(s) >= 1:
                    strt = dend_list.index(int(s[-1]))
                    somarootdist = Soma['soma root distance'][Soma['dendrite ID'] == s[0]].tolist()[0]
                    somaID = Soma['soma ID'][Soma['dendrite ID'] == s[0]].tolist()[0]
                else:
                    strt = dend_list.index(int(root))
                    s = [root]
                    somarootdist = []
                    somaID = 'inside'

                dendlen     = 0
                seg_len     = []
                for i in range(len(dend_list[strt:])-1):
                    v1, v2  = int(dend_list[strt+i]), int(dend_list[strt+i+1])
                    seg_len = np.linalg.norm( vertices[v1] - vertices[v2] )
                    dendlen += seg_len

                soma_dist.append({'dendrite ID': ID,
                                  'soma ID': somaID,
                                  'soma spot distance': dendlen,
                                  'soma root distance': somarootdist})

        somaDF= pd.DataFrame(soma_dist)
        allSoma.append({'file': filestem,
                       'dists': somaDF})
    return allSoma

def filter_Coordinates(DF,root_pix,delta):
    """
    Select pixel coordinates around root_pix with dimension 2*delta

    Parameters
    ----------
    DF : DataFrame (pandas)
        DataFrame containing coordinates labelled as 'X', 'Y' and 'Z'.
    root_pix : numpy array
       Contains the center pixel coordinates
    delta : numpy array
        contains the extension of the area around $root_pix [deltaX,deltaY,deltaZ].

    Returns
    -------
    None.

    """
    DF = DF[DF['X'] <= root_pix[0]+delta[0]]
    DF = DF[DF['X'] >= root_pix[0]-delta[0]]
    DF = DF[DF['Y'] <= root_pix[1]+delta[1]]
    DF = DF[DF['Y'] >= root_pix[1]-delta[1]]
    DF = DF[DF['Z'] <= root_pix[2]+delta[2]]
    DF = DF[DF['Z'] >= root_pix[2]-delta[2]]
    return DF



#filters spot data by size (and center point intensity)
def filter_Spot_Data(spots, size, intensity = []):
    """
    Filters spot data by size (and center point intensity)

    Parameters
    ----------
    spots : DataFrame (pandas)
        DataFrame containing data on PSD95 spots. Must atleast contain the columns ['Diameter X'] and Diameter['Z']
    size : numpy array (float)
        [X/Y Size, Z size]. All are lower cutoffs
    intensity : float, optional
        Lower bound for intensity values. The default is []. Requires that spot also contains column labelled ['Intensity Center']

    Returns
    -------
    filteredDF : DataFrame (pandas)
        DataFrame with spots larger than $size (and brighter then $intensity)
    """

    filteredDF = spots[(spots['Diameter X']>= size[0]) & (spots['Diameter Z']>= size[1])]
    if intensity:
        filteredDF= filteredDF[filteredDF['Intensity Center'] >= intensity]

    return filteredDF


#
def load_Soma_Points(SomaFileList, allData):
    """
    Creates dataFrame (pandas) containing the distance of each primary dendrite (soma ID)
    to the root (center of cell soma)

    Parameters
    ----------
    SomaFileList : TYPE
        DESCRIPTION.
    allData : TYPE
        DESCRIPTION.

    Returns
    -------
    AllSomaDF : TYPE
        DESCRIPTION.

    """
    AllSomaDF = []
    for file in SomaFileList:
        somaDF = pd.DataFrame()
        Positions      = pd.read_csv(file, header = 2)
        filestem       = file.stem.split('_new')[0]
        somaDF['file'] = filestem
        spots          = [f['spots'][['dendrite ID','dendrite X', 'dendrite Y', 'dendrite Z']]
                          for f in allData if f['file'].split('_new')[0] == filestem][0]
        all_min_dists = []
        for i in range(0, len(Positions)):
            xyz = spots[['dendrite ID','dendrite X', 'dendrite Y', 'dendrite Z']]
            modlist = xyz[['dendrite X', 'dendrite Y', 'dendrite Z']] - Positions.iloc[i][['Position X','Position Y','Position Z']].tolist()
            dists = np.linalg.norm(modlist, axis = 1)
            #dist2 = np.linalg.norm(data['vertices'] - Positions.iloc[i][['Position X','Position Y','Position Z']].tolist())
            min_dist = np.min(dists)
            min_dist_index = dists.tolist().index(min_dist)
      #      print(spots.iloc[min_dist_index])
            all_min_dists.append({
                'soma ID'           : Positions.iloc[i]['Name'],
                'dendrite soma dist': min_dist,
                'dendrite ID' : xyz.iloc[min_dist_index]['dendrite ID'],
                'dendrite X'  : xyz.iloc[min_dist_index]['dendrite X'],
                'dendrite Y'  : xyz.iloc[min_dist_index]['dendrite Y'],
                'dendrite Z'  : xyz.iloc[min_dist_index]['dendrite Z']
                })
        AllSomaDF.append({'file' : file,
                          'soma data' : pd.DataFrame(all_min_dists)
                               })
    return AllSomaDF


def add_Soma_Distances(DataList, SomaList, TreeList):
    """
    Adds the distance of the center of soma (root) to the start of the primary dendrites
    works directly on the DataFrame containing the distances (data) (DataList is list
    for all cells)

    Parameters
    ----------
    DataList : TYPE
        DESCRIPTION.
    SomaList : TYPE
        DESCRIPTION.
    TreeList : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for data in DataList:
        filestem = data['file']
        currsoma = [f for f in SomaList if f['file'].stem.split('_new')[0] == filestem][0]
        soma     = currsoma['soma data']
        T        = [f['Tree'] for f in TreeList if f['file'].split('_new')[0] == filestem][0]
        root     = [f['root'] for f in TreeList if f['file'].split('_new')[0] == filestem][0]
        vertices = [f['vertices'] for f in TreeList if f['file'].split('_new')[0] == filestem][0]

        soma_dist = []
        for sp in soma['dendrite ID']:
            dend_list = []
            if T.has_node(sp):
                dend_list   = nx.shortest_path( T , source=root, target=sp )
                dendlen     = 0
                seg_len     = []
                for i in range(len(dend_list)-1):
                    v1, v2  = int(dend_list[i]), int(dend_list[i+1])
                    seg_len = np.linalg.norm( vertices[v1] - vertices[v2] )
                    dendlen += seg_len
                soma_dist.append({'dendrite ID': sp, 'soma root distance': dendlen})
        #        print(filestem, sp,dendlen)
        currsoma['root'] = root
        currsoma['soma data'] = pd.merge(soma,pd.DataFrame(soma_dist), how='inner', on=['dendrite ID'] )


def get_Spots_on_Soma(imsfilelist, DataTree, filteredData,  delta = [50,50,50], PSFsize = [0.3,0.3,1.1]):
    """
    Find spots aassociated with the soma of the cell. Calculates upper and lower thresholds of
    intensity in the soma image. For a spot to be associated with the soma, the intensity at the
    spot position has to be inbetween the  two thresholds. Additionally, spots with a size smaller
    than the PSF are discarded

    Parameters
    ----------
    imsfilelist : list
        pathlib Path to IMARIS files.
    DataTree : list
        networkx Trees containing dendrite structure of the cell.
    filteredData : list
        DataFrame (pandas): spot data filtered with respect to background level and size
    delta : numpy array
        area (+-[dX,dY,dZ]) around center of soma used for finding spots on the surface. The default is [50,50,50]
    PSFsize : numpy array
        min size limits of PSF for rejecting small spots (below PSF size = high probablity for
        noise). The default is [0.3,0.3,1.1]


    Returns
    -------
    somaPSD_DF: DataFrame (pandas)
        DataFrame containing:
                    {'ID'              : networkx node,
                    'Position X'       : float (micrometer),
                    'Position Y'       : float (micrometer),
                    'Position Z'       : float (micrometer),
                    'intensity cell'   : float,
                    'spot intensity'   : float,
                    'spot diameter XY' : float (micrometer),
                    'spot diameter Z'  : float (micrometer),
                    }

    """
    collected_somaPSD_DF = []
    for imsfile in imsfilelist:
        somacount = 0
        rejected  = 0
        filestem = imsfile.stem.split('_new')[0]
        f             = h5py.File( imsfile, 'r')
        pixel_dim     = get_Data_Dimensions_in_pixel(f)
        image_dim     = get_Data_Dimensions(f)
        stack0       = f['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data'][()]
        stackPSD     = f['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 1']['Data'][()]
        T = [T for T in DataTree if T['file'].split('_new')[0] == filestem][0]

        #find center of mass of soma
        root_coord      = T['vertices'][T['root']]
        root_pix        = get_pixel_position_from_coordinates(root_coord,pixel_dim, image_dim)
        root_subpix     = get_subpixel_position_from_coordinates(root_coord,pixel_dim, image_dim)

        #correct center position for soma by calculating the center of mass
        # no modifications of region for com needed
        if root_pix[2]>delta[2]:
            # calculate center of mass around root coordinate
            com = np.array(ndimage.measurements.center_of_mass(stack0[root_pix[2]-delta[2]:root_pix[2]+delta[2],
                       root_pix[1]-delta[1]:root_pix[1]+delta[1],root_pix[0]-delta[0]:root_pix[0]+delta[0]]))
            old_root_pix = root_pix
            root_pix = np.array(root_subpix-np.array([com[2]-delta[2],com[1]-delta[1],com[0]-delta[0]]),dtype = 'int')
        # prevent region of com to be outside of image data (this is only necessary in z direction)
        else:
            dz = root_pix[2]-1
            # calculate center of mass around root coordinate
            com = np.array(ndimage.measurements.center_of_mass(stack0[root_pix[2]-dz:root_pix[2]+dz,root_pix[1]-delta[1]:root_pix[1]+delta[1],root_pix[0]-delta[0]:root_pix[0]+delta[0]]))
            old_root_pix = root_pix
            root_pix = np.array(root_subpix-np.array([com[2]-delta[2],com[1]-delta[1],com[0]-dz]),dtype = 'int')
            delta    = [delta[0],delta[1],dz]
        pix_shift = old_root_pix-root_pix


        ### define region around root
        colData         = [d for d in filteredData if d['file'] == filestem][0]
        spotCoord       = add_pixel_position_from_coordinates_to_DF(colData['spots'],pixel_dim,image_dim)
        filteredSpots   = filter_Coordinates(spotCoord,root_pix,delta)

        ### Background from PSD95 intensity (signal inPSD95 channel inside/around soma)
        #PSD_profile     = get_Zprofile(stackPSD, root_pix)
        micron          = np.array(image_dim.loc['width'][['X','Y','Z']])
        pixel           = np.array(pixel_dim.iloc[0][['X','Y','Z']])
        fact            = pixel/micron
        PSDxyz          = np.array(fact*[2,2,2]/2, dtype = 'int')
        # mean pixel background
        BG_PSD          = np.mean(stackPSD[root_pix[2]-PSDxyz[2]:root_pix[2]+PSDxyz[2],root_pix[1]-PSDxyz[1]:root_pix[1]+PSDxyz[1],root_pix[0]-PSDxyz[0]:root_pix[0]+PSDxyz[0]])
        # standard deviation of background
        BG_PSD_SD       = np.std(stackPSD[root_pix[2]-PSDxyz[2]:root_pix[2]+PSDxyz[2],root_pix[1]-PSDxyz[1]:root_pix[1]+PSDxyz[1],root_pix[0]-PSDxyz[0]:root_pix[0]+PSDxyz[0]])
        # define lower threshold
        BGPSD           = BG_PSD+3*BG_PSD_SD

        # thresholds from soma intensity
        root_profile    = get_Zprofile(stack0, root_pix)
        lower_thresh    = 1.0*np.max(root_profile)/2
        upper_thresh    = 1.0*np.max(root_profile)

        # count spots that are associated with the soma
        # criteria:
        # lower soma threshold < Intensity in soma channel at PSD95 position < upper soma threshold
        #
        uniqueZ = np.unique(filteredSpots['Z'])
        rejected = 0
        somacount = 0
        somaPSD = []
        for z in uniqueZ:
            #loop through spots in plane
            spots   = filteredSpots[filteredSpots['Z'] ==z]
            for s in spots.iterrows():
                s = s[1]
                # select spot by soma intensity thresholds (stack0)
                # if signal at PSD95 position in the soma channel is either too low
                # (not associated with soma) or too high (not at surface, probably inside),
                # the spots is rejected
                if (stack0[z,int(s['Y']),int(s['X'])]>= lower_thresh) & (stack0[z,int(s['Y']),int(s['X'])] <= upper_thresh):
                    somaPSD.append({'ID': s['ID'],
                                   'Position X'       : s['Position X'],
                                   'Position Y'       : s['Position Y'],
                                   'Position Z'       : s['Position Z'],
                                   'intensity cell'   : stack0[z,int(s['Y']),int(s['X'])],
                                   'spot intensity'   : s['Intensity Center'],
                                   'spot diameter XY' : s['Diameter X'],
                                   'spot diameter Z'  : s['Diameter Z'],
                                   })
                    # filter for spot size (larger than 300x300x1100nm) and intensity (above background)
                    if (s['Diameter X'] > PSFsize[0]) & (s['Diameter Z'] > PSFsize[2]) & (s['Intensity Center'] > BGPSD):
                        somacount += 1
                    # for all other cases (spot size too small) add to rejected list
                    else:
                        rejected  += 1

                   # z = int(z-root_pix[2]+delta[2])

                else:
                    rejected += 1
        #add soma spot counts to DataFrame
        collected_somaPSD_DF.append({'file'       : imsfile.stem,
                                     'somacount'  : somacount,
                                     'rejected'   : rejected,
                                     'SomaCounts' : pd.DataFrame(somaPSD)})

    return collected_somaPSD_DF

#


# create cummulative histograms of distancs as pandas.DataFrame()
def make_cummulative_DataFrame(SpotSomaDistances, SomaCounts, binsize = 1.0):
    """
    create histogram DataFrame of spot distancs as pandas.DataFrame()

    Parameters
    ----------
    SpotSomaDistances : list of dictionaries
        Contains the distances of all spots to the start of their primary dendrite.
    SomaCounts : TYPE
        DESCRIPTION.
    binsize : float, optional
        Bin size for histogram. The default is 1.0.

    Returns
    -------
    df : DataFrame (pandas)
        DataFrame of collected data
    hist_df : DataFrame (pandas)
        DataFrame of Histograms.

    """
    df = pd.DataFrame()
    # load dataframes
    for filestem in SomaCounts:
        D = [f for f in SpotSomaDistances if f['file'] == filestem][0]
        df1 = pd.DataFrame(D['dists']['soma spot distance'].to_list(), columns = {D['file']})
        df1 = df1.append([df1, pd.DataFrame(np.zeros(SomaCounts[filestem]), columns = {filestem})], ignore_index=True)
        df = pd.concat([df,df1], ignore_index=False, axis=1)

    # calculate bin centers for histogralm and plotting
    mbin = int(binsize*np.ceil(df.max().max()/binsize))
    bns = list(np.arange(0,mbin+binsize,binsize))
    w = binsize
    c = (np.array(bns[0:-1])+w/2);
    # creat histograms
    hist_df = pd.DataFrame()
    for col in df:
        s = pd.Series(df[col])
        s = s[~s.isna()]
        [hist_dist,b] = np.histogram(s, bins = bns, density=True)
        df_temp = pd.DataFrame(np.cumsum(hist_dist)*binsize, columns = {col})
        hist_df = pd.concat([hist_df,df_temp], ignore_index=False, axis=1)

    # calculate SEM and average values
    err = np.std(hist_df, axis = 1)
    sem = err/np.sqrt(len(SomaCounts))
    ave = np.mean(hist_df, axis = 1)

    # save in DatFrama and return
    hist_df = pd.concat([hist_df, pd.DataFrame(ave ,columns = {'average'})], ignore_index=False, axis=1)
    hist_df = pd.concat([pd.DataFrame(c, columns = {'bin center'}),hist_df], ignore_index=False, axis=1)
    hist_df = pd.concat([hist_df,pd.DataFrame(err,columns = {'STD'})], ignore_index=False, axis=1)
    hist_df = pd.concat([hist_df,pd.DataFrame(sem,columns = {'SEM'})], ignore_index=False, axis=1)
    return df, hist_df

# create cummulative histograms of distancs as pandas.DataFrame()
def make_histogram_DataFrame(SpotSomaDistances, SomaCounts, binsize = 10.0):
    """
    create histogram DataFrame of spot distancs as pandas.DataFrame()

    Parameters
    ----------
    SpotSomaDistances : list of dictionaries
        Contains the distances of all spots to the start of their primary dendrite.
    SomaCounts : TYPE
        DESCRIPTION.
    binsize : float, optional
        Bin size for histogram. The default is 10.0.

    Returns
    -------
    df : DataFrame (pandas)
        DataFrame of collected data
    hist_df : DataFrame (pandas)
        DataFrame of Histograms.

    """
    df = pd.DataFrame()
    # load dataframes
    for filestem in SomaCounts:
        D = [f for f in SpotSomaDistances if f['file'] == filestem][0]
        df1 = pd.DataFrame(D['dists']['soma spot distance'].to_list(), columns = {D['file']})
        df1 = df1.append([df1, pd.DataFrame(np.zeros(SomaCounts[filestem]), columns = {filestem})], ignore_index=True)
        df = pd.concat([df,df1], ignore_index=False, axis=1)

    # calculate bin centers for histogram and plotting
    mbin = int(binsize*np.ceil(df.max().max()/binsize))
    bns = list(np.arange(0,mbin+binsize,binsize))
    w = binsize
    c = (np.array(bns[0:-1])+w/2)

    # create histograms
    hist_df = pd.DataFrame()
    for col in df:
        s = pd.Series(df[col])
        # remove NaN from data
        s = s[~s.isna()]
        [hist_dist,b] = np.histogram(s, bins = bns, density=False)
        df_temp = pd.DataFrame(hist_dist, columns = {col})
        hist_df = pd.concat([hist_df,df_temp], ignore_index=False, axis=1)

    # calculate SEM and average values
    err = np.std(hist_df, axis = 1)
    sem = err/np.sqrt(len(SomaCounts))
    ave = np.mean(hist_df, axis = 1)

    # save in DatFrama and return
    hist_df = pd.concat([hist_df, pd.DataFrame(ave ,columns = {'average'})], ignore_index=False, axis=1)
    hist_df = pd.concat([pd.DataFrame(c, columns = {'bin center'}),hist_df], ignore_index=False, axis=1)
    hist_df = pd.concat([hist_df,pd.DataFrame(err,columns = {'STD'})], ignore_index=False, axis=1)
    hist_df = pd.concat([hist_df,pd.DataFrame(sem,columns = {'SEM'})], ignore_index=False, axis=1)
    return df, hist_df



def count_branching_points(data,collected_soma):
    """
    Counts branching points

    Parameters
    ----------
    data : list of dictionaries
        Contains the main data created by Analyze_Dendritic_Tree()
    collected_soma : list of DataFrames (pandas)
        DataFrames containing at least the IDs of the primary branching point nodes ['vertex ID'] (networkx node ID)

    Returns
    -------
    all_tips: list of dictionaries

    """
    print(data['file'])
    soma   = [x['soma'] for x in collected_soma if x['file'].split('_new')[0] == data['file'].split('_new')[0]][0]

    vertices = data['vertices']
    T = data['Tree']
    Tree_branch_points = [x for x in T.nodes() if T.out_degree(x)>1 and T.in_degree(x)==1]
    Tree_leaves = [x for x in T.nodes() if T.out_degree(x)==0 and T.in_degree(x)==1]
    short_tips = 0
    short_tip_ID_list = []
    all_tips = []
    for s in soma['vertex ID']:
        subT = nx.dfs_tree(T,s)
        subTree_brp = [x for x in subT.nodes() if subT.out_degree(x)>1 and subT.in_degree(x)==1]
        for l in Tree_leaves:
            if nx.has_path(T,s,l):
                for brp in Tree_branch_points:
                    #nodes from branch point to leave (mighht contain additional branch points)
                    if nx.has_path(T,brp,l):
                        node_list = nx.shortest_path( T, source = brp, target = l)
                        int_list = list_intersect(node_list,Tree_branch_points)
                        if len(int_list)==1:
                            length = 0

                            for j in range(len(node_list)-1):
                                v1, v2  = node_list[j], node_list[j+1]
                                point1, point2 = vertices[v1], vertices[v2]
                                seg_len = np.linalg.norm( point1 - point2 )
                                length  += seg_len

                            all_tips.append({'soma ID'    : s,
                                             'branch ID' : brp,
                                             'leaf ID'   : l,
                                             'length'    : length,
                                             'br pts primary dend':len(subTree_brp)
                                             })
                            if length <10.0:
                                short_tip_ID_list.append([s,brp,l,length])
                                short_tips += 1

    return all_tips

def get_Soma_distances(Data, SomaFiles):
    """
    Calculated hte distance of the start of the primary dendrites (position at soma)
    to the root (=center of soma) along the dendritic tree

    Parameters
    ----------
    Data : list of Dictionaries
        Data created by
    CSVfolder : TYPE
        DESCRIPTION.

    Returns
    -------
    collected_soma : TYPE
        DESCRIPTION.
    """
    collected_soma  = []
    for data in Data:
        file = data['file'][0:-4]
        print(file)
        if file[-4:]=='corr':
            file = file[0:-5]
        somafile = [x for x in SomaFiles if x.stem.split('_new')[0] == data['file'].split('_new')[0]][0]
        Positions = pd.read_csv(somafile, header = 2)
        print(Positions)
        all_min_dists = []
        for i in range(0, len(Positions)):
            dist = []
            for xyz in data['vertices']:
                dist.append(np.linalg.norm(xyz - Positions.iloc[i][['Position X','Position Y','Position Z']].tolist()))
            #dist2 = np.linalg.norm(data['vertices'] - Positions.iloc[i][['Position X','Position Y','Position Z']].tolist())
            min_dist = np.min(dist)
            min_dist_index = dist.index(min_dist)
            all_min_dists.append({
                'vertex ID': min_dist_index,
                'soma point':Positions.iloc[i]['Name']})
        all_min_dists = pd.DataFrame(all_min_dists)

        root = data['root']
        soma = []
        for sp in all_min_dists['vertex ID']:
            if sp != root:
                dend_list   = nx.shortest_path( data['Tree'], source=root, target=sp )
            #    d           = data['vertices'][dend_list]
                dendlen     = 0
                seg_len     = []
                for i in range(len(dend_list)-1):
                    v1, v2  = dend_list[i], dend_list[i+1]
                    seg_len = np.linalg.norm( data['vertices'][v1] - data['vertices'][v2] )
                    dendlen += seg_len
                if dendlen >2.0:
                    soma.append({'vertex ID': sp,
                                 'length': dendlen})
        soma = {'file': data['file'],
                'root': root,
                'soma': pd.DataFrame(soma)}
        collected_soma.append(soma)
       # print(soma)

    return collected_soma




# =============================================================================
### general custom functions
# =============================================================================

def load_Data(filename):
    """
    load data from pickle container

    Parameters
    ----------
    filename : pathlib path

    Returns
    -------
    data : undefined
        loads variables as saved in the container, no predefined TYPE
    """
    fpckl       = open(str(filename),'rb')
    loaded_data    = pickle.load(fpckl)
    fpckl.close()
    return loaded_data


def save_Data(filename, Data):
    """
    save data to pickle container

    Parameters
    ----------
    filename : pathlib path
    Data : any type of supported Data
        pandas DataFrames work very well
    """
    fpckl       = open(str(filename),'wb')
    pickle.dump(Data,fpckl)
    fpckl.close()


def save_Data_as_Excel(DataList, sortkey, savedir, filename):
    """
        Saves specific DataFrame columns or (dictionary entries) in Excel file columns. not a universal writer, requires key 'file'

    Parameters
    ----------
    DataList : List of DataFrames (pandas) or Dictionaries
        DataFrames to be saved. Each list entry is saved as single sheet
    sortkey : string
        name of DataFrame column/Dictionary key to save.
    savedir : pathlib path
        Saving directory.
    filename : string
        Filename to save the data.

    Returns
    -------
    None.

    """
    Excel_output = savedir / str(filename+'.xlsx')
    Excel_writer = pd.ExcelWriter(Excel_output)
    for data in DataList:
        data[sortkey].to_excel(Excel_writer, data['file'])
    Excel_writer.save()


def get_List_of_Files(dirName,ext = 'csv', recursive = True):
    """
    Returns list of files in directory (recursively if wanted)

    Parameters
    ----------
    dirName : string or pathlib  path
        Path to irectory to search.
    ext : string, optional
        Extension to filter files. The default is 'csv'.
    recursive : Boolean, optional
        Switch for recursive folder search. The default is True.

    Returns
    -------
    list of pathlinb paths
        list of files in folder with extension $ext

    """
    dirpath = Path(dirName)
    if recursive:
        searchstr = '**/*.'+ext
    else:
        searchstr = '*.'+ext

    files = dirpath.glob(searchstr)
    return list(files)


def filter_List_of_Files(files, discriminator):
    """
    Filters a list of filenames contianing specific string ($discriminator)

    Parameters
    ----------
    files : list of pathlib path
        List of filenames to filter.
    discriminator : string
        string used as filter criterium

    Returns
    -------
    list of pathlib path
        List of filtered filenames.

    """
    newfiles = []
    for file in files:
        if discriminator in file.name:
            newfiles.append(file)
    return list(newfiles)


def list_intersect(a, b):
    """
    intersection of two lists calculated using sets

    Parameters
    ----------
    a : list
    b : list

    Returns
    -------
    list
        intersection of a and b.

    """
    return list(set(a) & set(b))

def collect_Data_from_StatisticFiles(intensitylist, diameterlist, positionlist):
    """
    Combines data from different IMARIS statistic output files (.csv)
    into one DataFrame (pandas)

    Parameters
    ----------
    intensitylist : list of pathlib path
        contains the center intensity of the spots
    diameterlist : list of pathlib path
        contains the diameter of the spots
    positionlist : list of pathlib path
        contains the X,Y, Z coordinates (micrometer) of the spots

    Returns
    -------
    collectedIDs : list of dictionaries
        collectedIDs= [{'file' : string,
                        'spots'  : ['ID','Intensity Center','Diameter X',
                                    'Diameter Z', 'Position X', 'Position Y',
                                    'Position Z']}]
    """
    collectedIDs = []
    for file  in intensitylist:
        filestem = file.stem.split('_Intensity')[0]
        print(filestem)
        intfile         = file
        intensityDF     = pd.read_csv(intfile, header = 2)
        diamfile        = [f for f in diameterlist if filestem in f.stem][0]
        diameterDF      = pd.read_csv(diamfile, header = 2)
        positionfile    = [f for f in positionlist if filestem in f.stem][0]
        positionDF      = pd.read_csv(positionfile, header = 2)
        combined        = pd.merge(intensityDF, diameterDF, how='inner', on=['ID'])
        combined        = pd.merge(combined, positionDF, how='inner', on=['ID'])
        select          = combined[['ID','Intensity Center','Diameter X','Diameter Z',
                                    'Position X', 'Position Y', 'Position Z']].copy()

        collectedIDs.append({'file' : file,
                             'spots'  : select})
    return collectedIDs



# =============================================================================
### Start of script
# =============================================================================


#  Set up data folder and files

savefolder      = Path('/Path/to/folder/for/analyzed/Data/')
mainfolder      = Path('/Path/to/folder/with/IMARIS/files/') #*.ims

spotFolder      = Path( 'subfolder/to/IMARIS/spot/detection/Statistics' ) #.csv
somaFolder      = Path( 'subfolder/to/IMARIS/soma/position/files' ) #.csv
BGFolder        = Path( 'subfolder/to/background/intensity/iles' ) #.csv

#List of Immaris files and their root ID (networkx) [(string, int)]
Input = [('file1.ims',rootID1),
         ('file2.ims',rootID2))#,
         ...
         ]
# create day tag for save files
now = dt.now()
dt_string = now.strftime("%d-%m-%Y")

# =============================================================================
### Step 1: Dendritic structure and associated spots
# =============================================================================

###   a) create dendritic structure as a Tree (networkx) from IMARIS .ims file
DataTree = Analyze_Dendritic_Tree(Input, mainfolder)

## create list of all IMARIS (.ims) and Statistic (.csv) files for further analysis
imsfilelist    = get_List_of_Files(mainfolder, 'ims', True)
spotfilelist   = get_List_of_Files(mainfolder / spotFolder, 'csv', True)

#diameter of spots (micrometer)
diameterlist   = filter_List_of_Files(spotfilelist,'Diameter')
#position (X,Y,Z) of spots (micrometer)
positionlist   = filter_List_of_Files(spotfilelist,'Position')
#intensity of spot center (a.u.)
intensitylist  = filter_List_of_Files(spotfilelist,'Intensity_Center_Ch=2')

#position of start of primary dendrite on soma
somafilelist   = get_List_of_Files(mainfolder / somaFolder, 'csv',True)
somalist       = filter_List_of_Files(somafilelist,'Position')

headerlabel = 'String for plot heading and filenames - '

# collect spot Data in one variable
collectedData = collect_Data_from_StatisticFiles(intensitylist, diameterlist, positionlist)

### b) select spots detected by IMARIS by intensity and size

## filter the Data by size and above BG intensity (BG+3*SD))
# get signal intensity background from Fiji files (.csv)
BGfiles = get_List_of_Files( mainfolder / BGFolder,'csv',False)

#It might be necessary to adjust the seperator, depending on how the csv file is constructed
BGdata = pd.DataFrame()
for file in BGfiles:
    df      = pd.read_csv(file)
    if len(df.keys())==1:
        df      = pd.read_csv(file,sep='\s+')
    BGdata = pd.concat([BGdata,pd.DataFrame(df[['Mean','StdDev','Min','Max']].mean(),
                                        columns = {file.stem.split('new')[0][:-1]})],
                                        ignore_index = False, axis =1)

## filter the spots by size and intensity
# spots smaller than the PSF ([0.3,0.3,1.1], micrometer) are rejected
# spots with an intensity lower than the background+three times the sta,ndard deviationof the
# background are also rejected (Int > BG+3*SD[BG])
# Background was determined in Fiji by selecting regions without visible spots

PSFSize = [0.3,1.1] # x and y dimensions are the same
collectedFilteredData = []
for data in collectedData:
    filteredData=[]
    fData = pd.DataFrame()
    filestem = data['file'].stem.split('_new')[0]
    print(filestem)
    BG = BGdata.loc['Mean'][filestem]
    SD = BGdata.loc['StdDev'][filestem]
    filteredData = filter_Spot_Data(data['spots'],PSFSize, BG+3*SD).reset_index(drop=True)
    collectedFilteredData.append({'file': filestem,
                                  'spots': filteredData})


###   c) find PSD95 spots associated with dendrites (mindisty <200nm)) and calculate distances along dendrite for each spot

mindist = .2 #200nm minimum distance to dendrite surface
print('finding points closer than '+str(mindist)+' microns in cells...')

# calculate distance to start of primary dendrite
allDistances = get_dendrite_spot_distances(collectedFilteredData[0:2], DataTree, mindist)

#save data as pckl and excel files
save_Data(savefolder / str('allDistances_'+dt_string+'.pckl'),allDistances)
save_Data_as_Excel(allDistances,'spots',savefolder,'allData_'+dt_string)

###  d) calculate distance for dendrite segments

# load distance of start of primary dendrite (soma) to root (center of soma)
# AllSoma contains all distances of the spots to the soma and root
allSoma = load_Soma_Points(somalist[2:4], allDistances)

# add root-soma distance to calculated distance of spots
add_Soma_Distances(allDistances, allSoma, DataTree)

# =============================================================================
### Step 2: Soma and associated spots
# =============================================================================

# requires analyzed Tree structure (from Analyze_Dendritic_Tree),
# filtered Data (from filter_Spot_Data)
# background from Fijij files (BGYoung and BGAdult)

## set up files
# filename and initial center  of soma [px]
BGregion = {'filename1': [x1,y1],
            'filename2':[x2,y2]#,
            #...
            }

### a) Determine spots that are associated with the soma
# (requires loading the actual image data from .ims file via ndimage)and thresholds determined in Fiji

# setup
filteredData = collectedFilteredData
delta           = [50,50,20]


SomaSpots= get_Spots_on_Soma(imsfilelist[0:2], DataTree, filteredData,  delta = [50,50,50])


# setup excel file for saving
savefile = savefolder/ str('SomaCount_'+dt_string+'.xlsx')
writer = pd.ExcelWriter(savefile)
for Soma in SomaSpots:
    Soma['SomaCounts'].to_excel(writer, Soma['file'])

writer.save()


# =============================================================================
### Step 3: Analysis of spot and denderite data
# =============================================================================


###  a) Create histograms of spot and dendrite distances (including and excluding the soma)

## Soma counts for each cell, cells that have saturated soma are ignored
#List of Immaris files and their  soma count(from Step 2): {string : int}
Input = {'file1.ims': NumSpotsOnSoma1,
         'file2.ims': NumSpotsOnSoma2#,
         #...
         }

## create  histograms
allSpotSomaDistances = get_Spot_Soma_Distances(allDistances, allSoma, DataTree,[0.3,1.1])

df_cummulative, histogram_df_cummulative = make_cummulative_DataFrame(allSpotSomaDistances,SomaCounts,1.0)

df, histogram_df = make_histogram_DataFrame(allSpotSomaDistances,SomaCounts,10.0)


## save files as Excel
Excel_output = savefolder / str('all_cummulative_Histograms_'+dt_string+'.xlsx')
Excel_writer = pd.ExcelWriter(Excel_output)
histogram_df_cummulative.to_excel(Excel_writer, 'cummulative Histograms')
Excel_writer.save()

Excel_output = savefolder / str('all_Histograms_'+dt_string+'.xlsx')
Excel_writer = pd.ExcelWriter(Excel_output)
histogram_df.to_excel(Excel_writer, 'Histograms')
Excel_writer.save()

### b) Count branch points on primary dendrites

# distance of soma to primary dendrite starting points
collected_soma = get_Soma_distances(DataTree, somalist)

# count number of branch points for each primary dendrite
collected_tips = []
writer = pd.ExcelWriter(str(savefolder)+'Immature_Cells_branches_'+dt_string+'.xlsx')
collected_summary = []
for data in DataTree:
    all_tips = count_branching_points(data,collected_soma)
    tips_DF = pd.DataFrame(all_tips)
    tips_DF.to_excel(writer,data['file'])
    collected_tips.append(tips_DF)

writer.save()



