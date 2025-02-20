import os
import yaml

def distance(point1: list[float], point2: list[float]) -> float:
    """
    Square of the Euclidean distance between two points.
    
    Args: 
        - point1: (list[float]). List of coordinates of the first point.
        - point2: (list[float]). List of coordinates of the second point.

    Return:
        - float: Square of the Euclidean distance between point1 and point2.
    """

    # Check that the two points have the same dimension
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension.")
    
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) 

def majority_vote(neighbors: list[int]) -> int:
    """
    Computes the most frequent class label encountered in the list neighbors.

    Args:
        - neighbors (list[int]): List of class labels of the neighbors point.

    Return:
        - int: most frequent class label encountered in the list neighbors
    """
    # Creating a dictionary
    c=dict()

    for i in neighbors:
        if i in c:
            c[i]+=1
        else:
            c[i]=1
    
    return max(c, key=c.get)

def read_config(file) -> dict:
   """
    Reads a YAML configuration file and returns its contents as a dictionary.
    
    Args:
       - file (str): The name of the YAML configuration file (without the extension).
    
    Returns:
        - dict: A dictionary containing the key-value pairs from the YAML file.
   """
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs


def read_file(FileName):
    FileName = os.path.abspath(FileName)
    DataSet_points = []
    DataSet_labels = []

    with open(FileName, 'r') as f:

        for line in f:
            parsed_argument = line.split(",")

            point = list(map(float, parsed_argument[:-1]))

            label = parsed_argument[-1].strip()

            label = 1 if label == 'g' else 0

            DataSet_points.append(point)
            DataSet_labels.append(label)

    return DataSet_points, DataSet_labels

