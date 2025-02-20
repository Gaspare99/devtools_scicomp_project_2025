import pyclassify.utils as utils
from pyclassify.classifier import kNN
import pytest

def test_distance():
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
    vote_1=[1,1,1,1, 0, 1,1, 0, 1,1,1,1]

    assert utils.majority_vote(vote_1) == 1
    
def instance_KNN():
    with pytest.raises(TypeError):
        kNN(2.5)
    with pytest.raises(ValueError):
        kNN(-1)
    
    knn = kNN(3)
    assert knn.k == 3


test_distance()
test_majority_vote()
instance_KNN()
print("All tests passed!")


    