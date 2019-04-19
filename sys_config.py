### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.

import os

project_root = os.getcwd()
data_root = '/home/nhat/ACDC-dataset'
test_data_root = '/home/nhat/ACDC-test'
local_hostnames = ['e5-gpu-server']  # used to check if on cluster or not,
                                     # enter the name of your local machine

##################################################################################

log_root = os.path.join(project_root, 'acdc_logdir')
preproc_folder = os.path.join(project_root,'preproc_data')



##################################################################################
image_size = (212, 212)
target_resolution = (1.36719, 1.36719)
