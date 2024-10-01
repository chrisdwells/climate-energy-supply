CDF   ^   
      lon       lat       time       nb2             CDI       GClimate Data Interface version 1.5.5 (http://code.zmaw.de/projects/cdi)    Conventions       CF-1.4                                                                                                                                                                                                                                                             history      
�Fri Nov 02 15:56:08 2012: cdo -f nc copy MIROC-ESM-CHEM_rcp60_r1i1p1_tas_2006-2099.globmean.nc4 MIROC-ESM-CHEM_rcp60_r1i1p1_tas_2006-2099.globmean.nc
Thu Oct 25 12:27:29 2012: cdo -s selyear,2006/2099 /iplex/01/2011/isimip/inputdata/global.mean.uncorrected/MIROC-ESM-CHEM/MIROC-ESM-CHEM_rcp60_r1i1p1_tas_20000101-20991231.globmean.nc4 /iplex/01/2011/isimip/inputdata/global.mean.uncorrected.ISIsplit/MIROC-ESM-CHEM/MIROC-ESM-CHEM_rcp60_r1i1p1_tas_2006-2099.globmean.nc4
Tue Oct 23 13:07:17 2012: cdo -s -yearmean /iplex/01/2011/isimip/inputdata/global.mean.uncorrected/MIROC-ESM-CHEM/MIROC-ESM-CHEM_rcp60_r1i1p1_tas_20000101-20991231.globmean.nc4.fldmean /iplex/01/2011/isimip/inputdata/global.mean.uncorrected/MIROC-ESM-CHEM/MIROC-ESM-CHEM_rcp60_r1i1p1_tas_20000101-20991231.globmean.nc4
Tue Oct 23 12:51:51 2012: cdo -s -fldmean /iplex/01/2011/isimip/inputdata_unbced/MIROC-ESM-CHEM/MIROC-ESM-CHEM_rcp60_r1i1p1_tas_20000101-20991231_halfdeg.nc4 /iplex/01/2011/isimip/inputdata/global.mean.uncorrected/MIROC-ESM-CHEM/MIROC-ESM-CHEM_rcp60_r1i1p1_tas_20000101-20991231.globmean.nc4.fldmean
Fri Jul 06 20:32:29 2012: cdo -f nc4 -z zip copy /scratch/01/buechner/ISIMIP/models_int_MB/MIROC-ESM-CHEM_rcp60_r1i1p1_tas_20000101-20991231_halfdeg.nc /iplex/01/2011/isimip/inputdata_unbced/MIROC-ESM-CHEM/MIROC-ESM-CHEM_rcp60_r1i1p1_tas_20000101-20991231_halfdeg.nc4
Fri Jun 08 18:15:06 2012: cdo -s -P 2 -r -setreftime,1901-01-01,00:00,days -remapbil,/iplex/01/2011/isimip/data/crugrid.des -selyear,2000/2099 /scratch/01/buechner/ISIMIP/data/models_int/MIROC-ESM-CHEM/tas/rcp60/MIROC-ESM-CHEM_rcp60_tas_allTS.nc /iplex/01/2011/isimip/data/models_int_MB/MIROC-ESM-CHEM_rcp60_r1i1p1_tas_20000101-20991231_halfdeg.nc
Fri Jun 08 18:14:45 2012: cdo -s -P 2 -cat /scratch/01/buechner/ISIMIP/data/models_int/MIROC-ESM-CHEM/tas/historical/MIROC-ESM-CHEM_historical_tas_20000101-20051231.nc /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/rcp60/day/tas/MIROC-ESM-CHEM/r1i1p1/tas_day_MIROC-ESM-CHEM_rcp60_r1i1p1_20060101-21001231.nc /scratch/01/buechner/ISIMIP/data/models_int/MIROC-ESM-CHEM/tas/rcp60/MIROC-ESM-CHEM_rcp60_tas_allTS.nc
Fri Jun 08 17:53:46 2012: cdo -s -P 2 -selyear,2000/2005 /scratch/01/buechner/ISIMIP/data/models_int/MIROC-ESM-CHEM/tas/historical/MIROC-ESM-CHEM_historical_tas_allTS.nc /scratch/01/buechner/ISIMIP/data/models_int/MIROC-ESM-CHEM/tas/historical/MIROC-ESM-CHEM_historical_tas_20000101-20051231.nc
Fri Jun 08 17:42:34 2012: cdo -s -P 2 -cat /iplex/01/ipcc_ar5_pcmdi/pcmdi_data/historical/day/tas/MIROC-ESM-CHEM/r1i1p1/tas_day_MIROC-ESM-CHEM_historical_r1i1p1_18500101-20051231.nc /scratch/01/buechner/ISIMIP/data/models_int/MIROC-ESM-CHEM/tas/historical/MIROC-ESM-CHEM_historical_tas_allTS.nc
2011-10-21T02:51:28Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.      institution       �JAMSTEC (Japan Agency for Marine-Earth Science and Technology, Kanagawa, Japan), AORI (Atmosphere and Ocean Research Institute, The University of Tokyo, Chiba, Japan), and NIES (National Institute for Environmental Studies, Ibaraki, Japan)    institute_id      MIROC                                                                                                                                                                                                                                                              experiment_id         historical                                                                                                                                                                                                                                                         model_id      MIROC-ESM-CHEM                                                                                                                                                                                                                                                     forcing       GHG, SA, Oz, LU, Sl, Vl, MD, BC, OC (Ozone is predicted)                                                                                                                                                                                                           parent_experiment_id      piControl                                                                                                                                                                                                                                                          parent_experiment_rip         r1i1p1                                                                                                                                                                                                                                                             branch_time       @��        contact       Michio Kawamiya (kawamiya@jamstec.go.jp) and Toru Nozawa (nozawa@nies.go.jp)                                                                                                                                                                                       
references        Watanabe et al., : MIROC-ESM: model description and basic results of CMIP5-20c3m experiments, Geosci. Model Dev. Discuss., 4, 1063-1128, doi:10.5194/gmdd-4-1063-2011, 2011.                                                                                       initialization_method               physics_version             tracking_id       %19588900-bdb9-4410-85be-42cab093fefa       product       output                                                                                                                                                                                                                                                             
experiment        historical                                                                                                                                                                                                                                                         	frequency         day                                                                                                                                                                                                                                                                creation_date         2011-10-21T02:51:28Z                                                                                                                                                                                                                                               
project_id        CMIP5                                                                                                                                                                                                                                                              table_id      Table day (26 July 2011) f21c16b785432e6bd3f72e80f2cade49                                                                                                                                                                                                          title         MIROC-ESM-CHEM model output prepared for CMIP5 historical                                                                                                                                                                                                          parent_experiment         pre-industrial control                                                                                                                                                                                                                                             modeling_realm        atmos                                                                                                                                                                                                                                                              realization             cmor_version      2.7.1      CDO       HClimate Data Operators version 1.5.5 (http://code.zmaw.de/projects/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           &�   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           '   time               standard_name         time   bounds        	time_bnds      units         days since 1901-01-01 00:00:00     calendar      standard        '   	time_bnds                     units         days since 1901-01-01 00:00:00     calendar      standard        '   tas                       standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   original_name         T2     cell_methods      time: mean     history       �2011-10-21T02:51:28Z altered by CMOR: Treated scalar dimension: 'height'. 2011-10-21T02:51:28Z altered by CMOR: replaced missing value flag (-999) with standard missing value (1e+20). 2011-10-21T02:51:28Z altered by CMOR: Inverted axis: lat.      associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_MIROC-ESM-CHEM_historical_r0i0p0.nc areacella: areacella_fx_MIROC-ESM-CHEM_historical_r0i0p0.nc          '$                @��p    @��    @��    C��`@�    @��    @�     C���@�B�    @�     @�B�    C��k@�pp    @�B�    @�p�    C��@�    @�p�    @�     C��@�˰    @�     @���    C�Ϟ@��p    @���    @���    C���@�'    @���    @�'     C��K@�T�    @�'     @�T�    C�ֳ@�P    @�T�    @�`    C��^@�    @�`    @�     C�ܼ@�ݰ    @�     @���    C��@�P    @���    @�`    C��@�8�    @�`    @�9     C��C@�f�    @�9     @�f�    C���@�P    @�f�    @�`    C���@���    @�`    @��     C��<@��    @��     @��    C���@�P    @��    @�`    C���@�J�    @�`    @�K     C�<@�x�    @�K     @�x�    C�@�0    @�x�    @�@    C�B@���    @�@    @��     C�k@��    @��     @��    C�@�/0    @��    @�/@    C��@�\�    @�/@    @�\�    C�D@犐    @�\�    @犠    C��@�0    @犠    @�@    C��@���    @�@    @���    C���@�p    @���    @��    C��@�A0    @��    @�A@    C��@�n�    @�A@    @�n�    C��@�p    @�n�    @蜀    C�$@��    @蜀    @��     C�1�@���    @��     @���    C�9&@�%p    @���    @�%�    C�B[@�S    @�%�    @�S     C�K�@逰    @�S     @��    C�b�@�p    @��    @鮀    C�_�@��    @鮀    @��     C�_�@�	�    @��     @�	�    C�a�@�7P    @�	�    @�7`    C�]�@�e    @�7`    @�e     C�g�@꒰    @�e     @��    C�e�@��P    @��    @��`    C�`H@���    @��`    @��     C�pO@��    @��     @��    C�o�@�IP    @��    @�I`    C�sC@�v�    @�I`    @�w     C�j�@뤐    @�w     @뤠    C�z9@��P    @뤠    @��`    C�~+@���    @��`    @�      C���@�-�    @�      @�-�    C��u@�[0    @�-�    @�[@    C���@��    @�[@    @�     C��D@춐    @�     @춠    C��@��0    @춠    @��@    C��I@��    @��@    @��    C���@�?�    @��    @�?�    C��H@�m0    @�?�    @�m@    C���@��    @�m@    @��    C���@��p    @��    @�Ȁ    C���@��0    @�Ȁ    @��@    C���@�#�    @��@    @�#�    C��8@�Qp    @�#�    @�Q�    C��z@�    @�Q�    @�     C���@��    @�     @��    C��R@��p    @��    @�ڀ    C��;@�    @�ڀ    @�     C��.@�5�    @�     @�5�    C��E@�cp    @�5�    @�c�    C��,@�    @�c�    @�     C�۰@ﾰ    @�     @��    C��X@��P    @��    @��`    C��*@�    @��`    @�    C��@�#�    @�    @�#�    C� �@�:�    @�#�    @�:�    C��m@�Qx    @�:�    @�Q�    C��@�hX    @�Q�    @�h`    C� �@�(    @�h`    @�0    C� �@��    @�0    @�     C��@��    @�     @��    C��@�è    @��    @�ð    C�@��x    @�ð    @�ڀ    C�q@��H    @�ڀ    @��P    C�&5@�    @��P    @�     C��@��    @�     @�     C��@�5�    @�     @�5�    C��@�L�    @�5�    @�L�    C� �@�ch    @�L�    @�cp    C�*�@�zH    @�cp    @�zP    C�>@�    @�zP    @�     C�?X@��    @�     @��    C�=N@�    @��    @��    C�5m