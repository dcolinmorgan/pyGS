import os
import json
import numpy as np
import pickle
from pathlib import Path
import requests

class Exchange:
    def __init__(self):
        # Initialize any necessary properties here
        self.data = {}

    def save(self, obj, savepath='', fending='.pickle'):
        if not savepath:
            print('Save path does not exist')
            return

        if 'G' in obj.__dict__:
            obj.network = obj.name
            obj.A = np.asarray(obj.A).tolist()
            obj.G = np.asarray(obj.G).tolist()
            obj.nodes = list(obj.nodes)
            name = obj.name
            # Clean up obj before saving
            keys_to_remove = ['structure', 'name', 'nodes', 'params']
            for key in keys_to_remove:
                obj.__dict__.pop(key, None)
        else:
            print("Object does not have expected attributes.")
            return

        filename = os.path.join(savepath, f"{name}{fending}")
        if fending == '.json':
            with open(filename, 'w') as fp:
                json.dump(obj.__dict__, fp)
        elif fending in ['.ubj', '.pickle']:
            with open(filename, 'wb') as file:
                if fending == '.ubj':
                    # Suppose you have a function to serialize to UBJSON format
                    serialized_data = serialize_to_ubjson(obj.__dict__)
                    file.write(serialized_data)
                else:
                    pickle.dump(obj, file)

    def load(self, filepath):
        if not os.path.exists(filepath):
            print("File does not exist.")
            return None

        extension = os.path.splitext(filepath)[-1]
        with open(filepath, 'rb' if extension in ['.ubj', '.pickle'] else 'r') as file:
            if extension == '.json':
                return json.load(file)
            elif extension == '.ubj':
                return ubjson.load(file)
            elif extension == '.pickle':
                return pickle.load(file)

    # Example of a method to serialize to UBJSON, replace with actual implementation
    def serialize_to_ubjson(self, data):
        return b''  # Placeholder

# # Example usage
# ex = Exchange()
# obj = ex.load('path/to/data.json')  # Assuming data.json is a valid path
# if obj:
#     ex.save(obj, 'path/to/save/', '.json')
