import json

def save_to_file_json(data, file_name):
  with open(file_name, 'w') as f:
    json.dump(data, f)

def load_from_file_json(file_name):
  with open(file_name) as f:
    data = json.load(f)
  return data

def append_to_file_json(data, file_name):
  old_data = load_from_file_json(file_name)
  data = {**data, **old_data}
  save_to_file_json(data, file_name)

def create_file(file_name):
  with open(file_name, 'w') as _:
    pass