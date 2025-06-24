import json
import os

model_db_path = os.path.join('..', 'data', 'model_database.json')
with open(model_db_path, 'r') as f:
    data = json.load(f)
    
for model in data.keys():
    print(model) 