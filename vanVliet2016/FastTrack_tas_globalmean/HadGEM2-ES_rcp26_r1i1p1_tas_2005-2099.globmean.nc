CDF   _   
      lon       lat       time       nb2             CDI       GClimate Data Interface version 1.5.5 (http://code.zmaw.de/projects/cdi)    Conventions       CF-1.4                                                                                                                                                                                                                                                             history      �Fri Nov 02 15:52:24 2012: cdo -f nc copy HadGEM2-ES_rcp26_r1i1p1_tas_2005-2099.globmean.nc4 HadGEM2-ES_rcp26_r1i1p1_tas_2005-2099.globmean.nc
Thu Oct 25 12:27:27 2012: cdo -s selyear,2005/2099 /iplex/01/2011/isimip/inputdata/global.mean.uncorrected/HadGEM2-ES/HadGEM2-ES_rcp26_r1i1p1_tas_20000101-20991231.globmean.nc4 /iplex/01/2011/isimip/inputdata/global.mean.uncorrected.ISIsplit/HadGEM2-ES/HadGEM2-ES_rcp26_r1i1p1_tas_2005-2099.globmean.nc4
Tue Oct 23 12:39:21 2012: cdo -s -yearmean /iplex/01/2011/isimip/inputdata/global.mean.uncorrected/HadGEM2-ES/HadGEM2-ES_rcp26_r1i1p1_tas_20000101-20991231.globmean.nc4.fldmean /iplex/01/2011/isimip/inputdata/global.mean.uncorrected/HadGEM2-ES/HadGEM2-ES_rcp26_r1i1p1_tas_20000101-20991231.globmean.nc4
Tue Oct 23 12:22:05 2012: cdo -s -fldmean /iplex/01/2011/isimip/inputdata_unbced/HadGEM2-ES/HadGEM2-ES_rcp26_r1i1p1_tas_20000101-20991231_halfdeg.nc4 /iplex/01/2011/isimip/inputdata/global.mean.uncorrected/HadGEM2-ES/HadGEM2-ES_rcp26_r1i1p1_tas_20000101-20991231.globmean.nc4.fldmean
Fri Jul 06 18:28:30 2012: cdo -f nc4 -z zip copy /iplex/01/2011/isimip/inputdata_unbced/HadGEM2-ES/HadGEM2-ES_rcp26_r1i1p1_tas_20000101-20991231_halfdeg.nc /iplex/01/2011/isimip/inputdata_unbced/HadGEM2-ES/HadGEM2-ES_rcp26_r1i1p1_tas_20000101-20991231_halfdeg.nc4
Thu Jun 07 19:46:40 2012: cdo -s -P 2 -r -setreftime,1901-01-01,00:00,days -remapbil,/iplex/01/2011/isimip/data/crugrid.des -selyear,2000/2099 /scratch/01/buechner/ISIMIP/data/models_int/HadGEM2-ES/tas/rcp26/HadGEM2-ES_rcp26_tas_allTS.nc /iplex/01/2011/isimip/data/models_int_MB/HadGEM2-ES_rcp26_r1i1p1_tas_20000101-20991231_halfdeg.nc
Thu Jun 07 19:44:44 2012: cdo -s -P 2 -cat /scratch/01/buechner/ISIMIP/data/models_int/HadGEM2-ES/tas/historical/HadGEM2-ES_historical_tas_20000101-20051231.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/rcp26/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_rcp26_r1i1p1_20051201-20151130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/rcp26/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_rcp26_r1i1p1_20151201-20251130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/rcp26/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_rcp26_r1i1p1_20251201-20351130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/rcp26/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_rcp26_r1i1p1_20351201-20451130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/rcp26/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_rcp26_r1i1p1_20451201-20551130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/rcp26/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_rcp26_r1i1p1_20551201-20651130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/rcp26/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_rcp26_r1i1p1_20651201-20751130.nc
Thu Jun 07 19:44:40 2012: cdo -s -P 2 -selyear,2000/2005 /scratch/01/buechner/ISIMIP/data/models_int/HadGEM2-ES/tas/historical/HadGEM2-ES_historical_tas_allTS.nc /scratch/01/buechner/ISIMIP/data/models_int/HadGEM2-ES/tas/historical/HadGEM2-ES_historical_tas_20000101-20051231.nc
Thu Jun 07 19:34:17 2012: cdo -s -P 2 -cat /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/historical/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_historical_r1i1p1_18591201-18691130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/historical/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_historical_r1i1p1_18691201-18791130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/historical/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_historical_r1i1p1_18791201-18891130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/historical/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_historical_r1i1p1_18891201-18991130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/historical/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_historical_r1i1p1_18991201-19091130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/historical/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_historical_r1i1p1_19091201-19191130.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/historical/day/tas/HadGEM2-ES/r1i1p1/tas_day_HadGEM2-ES_historical_r1i1p1_19191201-19291130.nc
MOHC pp to CMOR/NetCDF convertor (version 1.5) 2010-11-22T15:34:21Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.   institution       aMet Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK, (http://www.metoffice.gov.uk)      institute_id      MOHC                                                                                                                                                                                                                                                               experiment_id         historical                                                                                                                                                                                                                                                         model_id      HadGEM2-ES                                                                                                                                                                                                                                                         forcing       GHG, SA, Oz, LU, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFCs)                                                                                                                                                                                                       parent_experiment_id      piControl                                                                                                                                                                                                                                                          parent_experiment_rip         r1i1p1                                                                                                                                                                                                                                                             branch_time                  contact       chris.d.jones@metoffice.gov.uk, michael.sanderson@metoffice.gov.uk                                                                                                                                                                                                 
references       �Bellouin N. et al, (2007) Improved representation of aerosols for HadGEM2. Meteorological Office Hadley Centre, Technical Note 73, March 2007; Collins W.J.  et al, (2008) Evaluation of the HadGEM2 model. Meteorological Office Hadley Centre, Technical Note 74,; Johns T.C. et al, (2006) The new Hadley Centre climate model HadGEM1: Evaluation of coupled simulations. Journal of Climate, American Meteorological Society, Vol. 19, No. 7, pages 1327-1353.; Martin G.M. et al, (2006) The physical properties of the atmosphere in the new Hadley Centre Global Environmental Model, HadGEM1 - Part 1: Model description and global climatology. Journal of Climate, American Meteorological Society, Vol. 19, No.7, pages 1274-1301.; Ringer M.A. et al, (2006) The physical properties of the atmosphere in the new Hadley Centre Global Environmental Model, HadGEM1 - Part 2: Aspects of variability and regional climate. Journal of Climate, American Meteorological Society, Vol. 19, No. 7, pages 1302-1326.      initialization_method               physics_version             tracking_id       %13e35a66-3a1d-44bb-be37-e24b90c0f0f8       mo_runid      ajhoh                                                                                                                                                                                                                                                              product       output                                                                                                                                                                                                                                                             
experiment        historical                                                                                                                                                                                                                                                         	frequency         day                                                                                                                                                                                                                                                                creation_date         2010-11-22T15:34:22Z                                                                                                                                                                                                                                               
project_id        CMIP5                                                                                                                                                                                                                                                              table_id      Table day (12 November 2010) 53fa6f63b86081d1c644183416239052                                                                                                                                                                                                      title         HadGEM2-ES model output prepared for CMIP5 historical                                                                                                                                                                                                              parent_experiment         pre-industrial control                                                                                                                                                                                                                                             modeling_realm        atmos                                                                                                                                                                                                                                                              realization             cmor_version      2.5.0      CDO       HClimate Data Operators version 1.5.5 (http://code.zmaw.de/projects/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           /P   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           /X   time               standard_name         time   bounds        	time_bnds      units         days since 1901-01-01 00:00:00     calendar      360_day         /`   	time_bnds                     units         days since 1901-01-01 00:00:00     calendar      360_day         /h   tas                    	   standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   comment       <daily-mean near-surface (usually, 2 meter) air temperature.    original_name         mo: m01s03i236     cell_methods      time: mean     history       �2010-11-22T15:34:22Z altered by CMOR: Treated scalar dimension: 'height'. 2010-11-22T15:34:22Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).    associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_HadGEM2-ES_historical_r0i0p0.nc areacella: areacella_fx_HadGEM2-ES_historical_r0i0p0.nc          /x                @�t�    @�H     @�u     C��9@��    @�u     @�     C���@���    @�     @��     C��_@���    @��     @��     C��?@�(�    @��     @�)     C��@�U�    @�)     @�V     C��@��    @�V     @�     C���@��    @�     @�     C���@���    @�     @��     C�ܢ@�	�    @��     @�
     C��R@�6�    @�
     @�7     C��@�c�    @�7     @�d     C�Տ@��    @�d     @�     C��@@��    @�     @�     C���@���    @�     @��     C��+@��    @��     @�     C��@�D�    @�     @�E     C�v@�q�    @�E     @�r     C�)�@��    @�r     @�     C�(U@���    @�     @��     C�(�@���    @��     @��     C�$7@�%�    @��     @�&     C�C@�R�    @�&     @�S     C�
@��    @�S     @�     C�(-@��    @�     @�     C�0.@���    @�     @��     C�<U@��    @��     @�     C�5
@�3�    @�     @�4     C�C�@�`�    @�4     @�a     C�?@��    @�a     @�     C�@�@��    @�     @�     C�K�@���    @�     @��     C�5@��    @��     @�     C�8�@�A�    @�     @�B     C��@�n�    @�B     @�o     C�:@��    @�o     @�     C�$�@���    @�     @��     C�(1@���    @��     @��     C�-5@�"�    @��     @�#     C�2�@�O�    @�#     @�P     C�>
@�|�    @�P     @�}     C�=�@��    @�}     @�     C�I<@���    @�     @��     C�]C@��    @��     @�     C�`�@�0�    @�     @�1     C�l@�]�    @�1     @�^     C�]O@��    @�^     @�     C�Z�@��    @�     @�     C�Z5@���    @�     @��     C�Q�@��    @��     @�     C�O8@�>�    @�     @�?     C�Ag@�k�    @�?     @�l     C�S�@��    @�l     @�     C�[�@���    @�     @��     C�O�@���    @��     @��     C�Qu@��    @��     @�      C�L�@�L�    @�      @�M     C�R�@�y�    @�M     @�z     C�QW@��    @�z     @�     C�K�@���    @�     @��     C�5�@� �    @��     @�     C�D�@�-�    @�     @�.     C�U�@�Z�    @�.     @�[     C�Z�@��    @�[     @�     C�M�@���    @�     @��     C�V�@���    @��     @��     C�E�@��    @��     @�     C�M@�;�    @�     @�<     C�E8@�h�    @�<     @�i     C�R�@��    @�i     @�     C�G�@���    @�     @��     C�8�@���    @��     @��     C�G�@��    @��     @�     C�5�@�I�    @�     @�J     C�Cr@�v�    @�J     @�w     C�J�@��    @�w     @�     C�:�@���    @�     @��     C�4�@���    @��     @��     C�8�@�x    @��     @��    C�L�@�+�    @��    @�,     C�C@�Bx    @�,     @�B�    C�>�@�X�    @�B�    @�Y     C�J�@�ox    @�Y     @�o�    C�G�@���    @�o�    @��     C�MV@�x    @��     @�    C�HR@��    @�    @�     C�U$@��x    @�     @�ɀ    C�H@���    @�ɀ    @��     C�:m@��x    @��     @���    C�F@��    @���    @�     C�8�@�#x    @�     @�#�    C�<�@�9�    @�#�    @�:     C�A}@�Px    @�:     @�P�    C�2�@�f�    @�P�    @�g     C�2;@�}x    @�g     @�}�    C�J�