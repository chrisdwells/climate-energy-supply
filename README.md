Main FRIDA repository: https://github.com/metno/WorldTransFRIDA 
This repository creates functions to estimate the changes in hydroelectric and thermoelectric energy supply under climate change.
The scripts should be ran in order, in the scripts/ folder. 
Information on each is given in the code.

It uses data from Van Vliet et al 2016 (https://www.nature.com/articles/nclimate2903). The 00 script downloads ISIMIP2b climate data; the processed data 
from this is in data/processed/globalmean_annual, so script 01 will run without this step.