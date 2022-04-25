import os
import shutil
from os.path import join, exists


base_dir = None
files = os.listdir(base_dir)

new_dir = base_dir + '_copy'
if not exists(new_dir):
    os.makedirs(new_dir)

for f in files:
    old_addr = join(base_dir, f)
    new_addr = join(new_dir, f)
    shutil.copy(old_addr, new_addr)
        
    f2 = f.replace('SDP', 'DPM')
    new_addr2 = join(new_dir, f2)
    shutil.copy(old_addr, new_addr2)

    f3 = f.replace('SDP', 'FRCNN')
    new_addr3 = join(new_dir, f3)
    shutil.copy(old_addr, new_addr3)
