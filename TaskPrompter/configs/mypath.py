import os
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]

# db_root = '/pscratch/sd/h/hwchen/datasets'
db_root = '/projectnb/ivc-ml/sunxm/datasets'

db_names = {'PASCALContext': 'PASCALContext', 'NYUD_MT': 'NYUDv2', 'Cityscapes3D': 'Cityscapes', 'CityScapes_MT': 'cityscapes'}
db_paths = {}
for database, db_pa in db_names.items():
    db_paths[database] = os.path.join(db_root, db_pa)
