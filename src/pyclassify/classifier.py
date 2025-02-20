from . import utils

class kNN:
    def __init__(self, k=3):
        if not (isinstance(k, int)):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        
        self.k = k

    def _get_k_nearest_neighbors(self, X, y, x):

        distances = [(utils.distance(x, x_data), y_data) for x_data, y_data in zip(X, y)]
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        neighbors=[i for _ , i in neighbors]
        return neighbors
    
    def __call__(self, data, new_points):
        classified_vector = []
        for point in new_points:
            neighbors = self._get_k_nearest_neighbors(data[0], data[1], point)
            point_class = utils.majority_vote(neighbors)
            classified_vector.append(point_class)

        return classified_vector
    
#obj=kNN(3)