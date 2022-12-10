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

instance_dir = "data_runthis/200_50/"
instances = os.listdir(instance_dir)
urls = []
for i in range(len(instances)): # Loop for getting URLs
    urls.append(instance_dir + instances[i]) # Get the "Raw" link
    if instances[i] == '.ipynb_checkpoints':
        urls.pop()


batch.run_compare_GA(urls, instances)
print("I am done")


