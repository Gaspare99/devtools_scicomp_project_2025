from . import utils

class kNN:
    """
    In statistics, the k-nearest neighbors algorithm (k-NN) is a non-parametric supervised learning method. It was first developed by
    Evelyn Fix and Joseph Hodges in 1951, and later expanded by Thomas Cover. Most often, it is used for classification, as a k-NN 
    classifier, the output of which is a class membership. An object is classified by a plurality vote of its neighbors, with the 
    object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). 

    Source Wikipedia 
    https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
    """
    def __init__(self, k: int):
        """
        Class initializer. 

        Args: 
            - k (int): number of clusters you want to divide the dataset in.
        """
        if not (isinstance(k, int)):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        
        self.k = k

    def _get_k_nearest_neighbors(self, X: list[list[float]], y: list[int], x:  list[float])->list[int]:
        """
        Computes the class labels of the k nearest neighboors.

        Args:
            - X (list[list[float]]): Training DataSets (points). List of the coordinates of the points used for training
            - y (list[int]): Training DataSets (class labes). Class label that corrispond to the point of the training DataSet
            - x (list[float]): New point to be classsified.

        Returns:
            - list[int]: A list of class labels corresponding to the k-nearest neighbors.
        """

        distances = [(utils.distance(x, x_data), y_data) for x_data, y_data in zip(X, y)]
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        neighbors=[i for _ , i in neighbors]
        return neighbors
    
    def __call__(self, data: tuple[list[list[float]], list[int]], new_points: list[list[float]])-> list[int]:
        """
        Computes the class labels to which the new_points most likely belonging to, proving a certain data set.

         Args:
        - data (tuple[list[list[float]], list[int]]):
            - X (list[list[float]]): First entry of the tuple. List of training data points .
            - y (list[int]): Second argument. A list of class labels corresponding to the training data.
        - new_points (list[list[float]]): Point to classify.
        
        Returns:
        - list[int]: class labels to which the new_points most likely belonging to.
                 
        """
        classified_vector = []
        for point in new_points:
            neighbors = self._get_k_nearest_neighbors(data[0], data[1], point)
            point_class = utils.majority_vote(neighbors)
            classified_vector.append(point_class)

        return classified_vector
    
