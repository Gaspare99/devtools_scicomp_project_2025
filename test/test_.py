import pyclassify.utils as utils
from pyclassify.classifier import kNN
import pytest

def test_distance():
    """
    Test that the distance function satisfies all the properties that definte it
    (symmetry, positive definite, triangular inequalities etc.)
    """
    x=[1,2,3]
    y=[2,3,4] 
    z=[]
    for i, j in zip(x, y):
        z.append(i+j)
    zeros=[0]*len(x)
    assert utils.distance(x, x) == 0
    assert utils.distance(zeros, x) == 14
    assert utils.distance(zeros, x) > 0
    assert utils.distance(x, y) == utils.distance(y, x)
    print(utils.distance(zeros, x))
    print(utils.distance(zeros, y))
    print(utils.distance(zeros, z))
    print(utils.distance(zeros, y) + utils.distance(zeros, x))

    assert utils.distance(zeros, z) ** 0.5 < (utils.distance(x, zeros)**0.5 + utils.distance(y, zeros) **0.5) 



def test_majority_vote():
    """
    Tests to verify the correct implementation of the majority majority_vote function.
    Given the class labels of the neighboor points, returns the class labels that occures most often.
    """
    vote_1=[1,1,1,1, 0, 1,1, 0, 1,1,1,1]
    assert utils.majority_vote(vote_1) == 1

    vote_2=[0 ,0, 0, 1, 1, 0, 1, 1, 0, 0 , 0, 0]
    assert utils.majority_vote(vote_2) == 0

    
    
def test_instance_KNN():
    """
    Test the correct instantiation of the KNN class
    """
    with pytest.raises(TypeError):
        kNN(2.5)
    with pytest.raises(ValueError):
        kNN(-1)
    with pytest.raises(TypeError):
        kNN("String Argument")

    with pytest.raises(ValueError):
        kNN(1, backend="Random name")
    
    knn = kNN(3)
    assert knn.k == 3

    assert isinstance(knn, kNN)  

    knn2 = kNN(3, backend='plain')
    assert knn2.backend == 'plain'
    knn3 = kNN(3, backend='numpy')
    assert knn3.backend == 'numpy'



test_distance()
test_majority_vote()
test_instance_KNN()



    