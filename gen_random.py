import sys
sys.path.append("..")
import batch_run_functions as batch

# change this directory below:
instance_dir = "test_run/"
# instance_dir = "/data/100s/"
# instance_dir = "/data/200s/"
# instance_dir = "/data/300s/"

sample_number = 500  # for test
# Below is the number we should run, uncomment it
# sample_number = 500000
batch.run_random_shit(instance_dir, sample_number)

# %%
