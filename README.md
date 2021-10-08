# Developmental emergence of two-stage nonlinear synaptic integration in cerebellar interneurons
This script is used to analyse IMARIS data from two-photon imaging (PSD95-venus and Alexa594 cell body)

authors: 
Célia Biane, Florian Rückerl, Therese Abrahamsson, Cécile Saint-Cloment,  Jean Mariani , Ryuichi Shigemoto, David A. DiGregorio, Rachel M. Sherrard, and Laurence Cathala

author of the script:
Florian Rückerl, Oct 2021

 This is an outline for analyzing the data, it will need to be adapted for individual experiments. Data is saved as tables in excel format and/or as  .pckl files. Details of our methods can befound in the above mentioned article (eLife, link pending)

=============================================================================

Used  programming environment:

Python 3.7.6
conda 4.10.3
spyder 4.0.1



Used module versions:
 h5py        2.10.0
 networkx    2.5.1
 pandas      1.2.4
 scipy       1.6.2
 matplotlib  3.3.4
 numpy       1.20.2


=============================================================================

## Outline of the analysis steps in this script:

- Step 1: Dendritic structure and associated spots
   a) create dendritic structure as a Tree (networkx) from IMARIS 9.2 .ims file
   b) select spots detected by IMARIS by intensity and size
   c) find PSD95 spots associated with dendrites (mindist < 200nm)) and calculate distances along dendrite for each spot
   d) calculate distance for dendrite segments

- Step 2: Soma and associated spots
   a) Determine spots that are associated with the soma (requires loading the  actual image data from .ims file via ndimage)and thresholds determined in Fiji

- Step 3: Analysis of spot and dendrite data
   a) Create histograms of spot and dendrite distances (including and excluding the soma)
   b) Count branch points on primary dendrites


