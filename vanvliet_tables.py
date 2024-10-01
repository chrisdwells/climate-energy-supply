import pandas as pd



df_2 = pd.read_csv('Yalew2020/41560_2020_664_MOESM2_ESM.csv')
df_3 = pd.read_csv('Yalew2020/41560_2020_664_MOESM3_ESM.csv', 
                   encoding='unicode_escape')


df_2_thermal_global = df_2[(df_2['Category'] == 'Thermal') & 
                     (df_2['Region_larger'] == 'Global')]


df_3_thermal_global = df_3[(df_3['Category'] == 'Thermal') & 
                     (df_3['Scale'] == 'Global')]