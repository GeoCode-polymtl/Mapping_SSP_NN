# Mapping Subsea Permafrost using deep learning

This repository provides the code and results in "Mapping subsea permafrost distribution in the Beaufort Sea with marine 
seismic and deep learning" submitted to JGR: Solid Earth

The repository contains 4 folders aiming to store the required information (CheckpointsTL, DataPreprocessed, InvertedModels 
and SSPInterpretation) and 1 code folder. The data folders are as follows::

* Checkpoints: Store the last checkpoint on the TL methodology
* Datapreprocessed: Store de seismic lines aranged in CMPs. The subfolder CMPs_coords saves the location files of the 
lines and the subfolder MultiInput save the seismic line data transformed in the 4 input domains
* InvertedModels: Store the inverted models after evaluating the 64 NN in the subfolder TL and the average of the Vp model
in the subfolder VP_models
* SSPInterpretation: Store the inverted velocity models after applying the defined thresholds

The coding folder contain the scrpts necessary for obtaining the results from the TL methodology applied to the seismic 
data. There are two main files to consider:

* Evaluate.py: Read the information on the Datapreprocessed folder and generate the MultiInput file if it does not 
exist. It also generates the output file in InvertedModels/TL subfolder
* Sections_100mIsobath.py: Generate the interpeted sections of permafrost distribution in the sesimic lines and store 
them in the folder SSPInterpretation. In addition, it saves the Average Vp velocity models in the subfolder 
InvertedModels/VP_models

The requiements for running the scripts are summarized in the file requirements.txt. Note that the package GeoFlow is 
available in https://github.com/gfabieno/GeoFlow