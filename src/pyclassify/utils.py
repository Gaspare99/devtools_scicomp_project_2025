import os
import yaml

def distance(point1: list, point2: list):
    """Euclidean distance between two points."""
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension.")
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) 

def majority_vote(neighbors):
    c=dict()

    for i in neighbors:
        if i in c:
            c[i]+=1
        else:
            c[i]=1
    
    return max(c, key=c.get)

def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs