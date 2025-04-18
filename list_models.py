import json

with open('model_database.json', 'r') as f:
    data = json.load(f)
    
for model in data.keys():
    print(model) 