# %load_ext autoreload
# %autoreload 2
# !pip install deap
# !git clone https://github.com/jekim526/IE-541-Project.git
# !git --git-dir=/content/IE-541-Project/.git pull

import os
import sys
sys.path.append("..")
# sys.path.append('/content/IE-541-Project')
import batch_run_functions as batch

# instances = os.listdir("/content/test_run")
instances = os.listdir("test_run")
urls = []
for i in range(len(instances)): # Loop for getting URLs
    urls.append('https://raw.githubusercontent.com/jekim526/IE-541-Project/main/test_run/'+instances[i]) # Get the "Raw" link
    if instances[i] == '.ipynb_checkpoints':
        urls.pop()

batch.run_compare_GA(urls, instances)
print("I am done")
#%%
