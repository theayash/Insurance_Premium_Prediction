import json
with open("config.json", "r") as r:
    param = json.load(r)['params']

prediction_file = param['upload_location']
print(prediction_file)