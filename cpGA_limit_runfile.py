# %load_ext autoreload
# %autoreload 2
# !pip install deap
# !git clone https://github.com/jekim526/IE-541-Project.git
# !git --git-dir=/content/IE-541-Project/.git pull
#
import os
import sys
import time

sys.path.append("..")
# sys.path.append('/content/IE-541-Project')
import batch_run_functions as batch

"""  Settings """
instance_dir = "data_limit_test_run/1/"
instances = os.listdir(instance_dir)
urls = []
for i in range(len(instances)):  # Loop for getting URLs
    urls.append(instance_dir + instances[i])  # Get the "Raw" link
    if instances[i] == '.ipynb_checkpoints':
        urls.pop()

# run_compare_GA_resultOnly(urls, instances, GA_type, objs, knapsack_nums, t_gen, b_gen, tail=""):
GA_types = ['base', 'tugba']
objs = [1, 3]
knapsack_nums = [3, 5, 10]
#t_gen = 5
t_gen = 1000
#b_gen = 50
b_gen = 10000

""" running """
batch.run_compare_GA_genlog(urls, instances, GA_types, objs, knapsack_nums, t_gen, b_gen)
print("I am done")
