import parameter_experiment as pe
import sys
import time

sys.path.append("..")
sys.path.append('/content/IE-541-Project')

# url = 'https://raw.githubusercontent.com/jekim526/IE-541-Project/main/data/Z_r_100_25_1.csv' # Get the "Raw" link
# evolution_specify_parameters = (0.7, 0.5, 500, 500)
# print('parameters: ')
# print(evolution_specify_parameters)
# print('mean_objfValue: ')
# # def search_parameter(url, evolution_specify_parameters, ga_type, rounds = 10, popsize = 100):
# print(pe.search_parameter(url, evolution_specify_parameters, 1, 5, 200))

url = 'https://raw.githubusercontent.com/jekim526/IE-541-Project/main/data/Z_r_100_50_1.csv'  # Get the "Raw" link
evolution_specify_parameters = (0.7, 0.3, 1000, 10000)
print('parameters: ')
print(evolution_specify_parameters)
print('mean_objfValue: ')

time_start = time.time()
# def search_parameter(url, evolution_specify_parameters, ga_type, rounds = 10, popsize = 100):
print(pe.search_parameter(url, evolution_specify_parameters, 1, 3, 50))
time_end = time.time()
print('time cost', time_end - time_start, 's')
# %%
