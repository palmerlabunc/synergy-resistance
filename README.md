# Relationship between synergy and resistance in combination cancer therapy

This repository includes all of the code used to run the simulations of the relationship between synergy resistance and generate the associated figures in the article. Below is a summary of each script. 

* `checkerboard.py` simulates a checkerboard as a specified level of synergy and plots the relationship between excess bliss and fitness benefit for each point in the checkboard. 
* `increasing synergy.py` simulates checkerboards at multiple levels of synergy and plots the relationship between levels of synergy and median fitness benefit across the checkerboard.
* `heatmaps.py` simulates checkerboards at multiple levels of synergy and plots heatmaps of activity, excess bliss, and fitness benefit across each checkerboard
* `collateral effects.py` adds collateral effects including collateral sensitivity into the model and plots the fitness benefit across varying levels of synergy and collateral effect.  
* `isobolograms.py` generates representative isobologram plots for three different levels of synergy 

