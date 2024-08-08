"""
Generate the train test and validation folds for n-fold cross validations. Allows for a non-even splits of train test,
and validation eg: 60% 10% 30%
"""

# 3rd party
import yaml
from pathlib import Path
import numpy as np

data_dir = Path('/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/')
test_folds = 10
val_folds = 4
paths = [[fp for fp in sorted(Path(f'/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/tfrecords/eval').iterdir())]
         for num in range(5)]
print(len(paths))

cv_list = []
for i, sub_paths in enumerate(paths):

    cv_iter = {dataset: None for dataset in ['train', 'test']}
    print(sub_paths[i])
    cv_iter['test'] = [sub_paths[test_folds*i+j] for j in range(test_folds)]
    remaining = [fp_n for fp_n in sorted(sub_paths) if (fp_n not in cv_iter['test'])]
    rand_val_folds = np.random.choice(range(len(remaining)), val_folds, replace=False)
    cv_iter['val'] = [remaining[j] for j in rand_val_folds]
    no_test_fps = [fp_n for fp_n in sorted(sub_paths) if (fp_n not in cv_iter['test'] and fp_n not in cv_iter['val'])]
    print(no_test_fps)
    cv_iter['train'] = no_test_fps
    cv_list.append(cv_iter)

with open(f'/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/tfrecords/cv_folds_tfrecords.yaml', 'w+') as file:
    yaml.dump(cv_list, file, sort_keys=False)






#%%
#
# with(open('/Users/agiri1/Desktop/ExoBD_Datasets/shard_tables/cv_folds.yaml', 'r')) as file:
#     cv_folds = yaml.unsafe_load(file)