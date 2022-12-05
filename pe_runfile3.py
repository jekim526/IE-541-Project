import parameter_experiment as pe
import sys
sys.path.append("..")
sys.path.append('/content/IE-541-Project')

# url = 'https://raw.githubusercontent.com/jekim526/IE-541-Project/main/data/Z_r_100_25_1.csv' # Get the "Raw" link
# evolution_specify_parameters = (0.7, 0.3, 500, 500)
# print('parameters: ')
# print(evolution_specify_parameters)
# print('mean_objfValue: ')
# # def search_parameter(url, evolution_specify_parameters, ga_type, rounds = 10, popsize = 100):
# print(pe.search_parameter(url, evolution_specify_parameters, 1, 5, 50))


###------------------------------------------------------
url = 'https://raw.githubusercontent.com/jekim526/IE-541-Project/main/data/Z_r_100_50_1.csv'
evolution_specify_parameters = (0.7, 0.5, 10000, 10000)
print('parameters: ')
print(evolution_specify_parameters)
print('mean_objfValue: ')

import time
time_start=time.time()
print(pe.search_parameter(url, evolution_specify_parameters, 0, 3, 100))
time_end=time.time()
print('time cost',time_end-time_start,'s')

#%%
