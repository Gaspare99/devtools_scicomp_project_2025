from pyclassify.classifier import kNN
from pyclassify.utils import  read_config, read_file
import argparse
import yaml
import numpy as np


parser = argparse.ArgumentParser(description="Run kNN classification.")
parser.add_argument("--config", type=str, help="Path to config file.", required=True)

args = parser.parse_args()
config = read_config(args.config)
k = config['k']
backend = config['beckend']
dataset = config['dataset']
print(backend)

classification= kNN(k, backend=backend)



print(f"k: {k}, dataset: {dataset}")

X, y = read_file(dataset)

N=len(y)
index = np.arange(N)
np.random.shuffle(index)
X = [X[i] for i in index]
y = [y[i] for i in index]
i=int(N*0.2)
train_data=(X[:i], y[:i])
test_data=X[i:]
test_y=y[i:]

y_class=classification(train_data, test_data)
for i, j in zip(y_class, test_y):
    print(f"Predicted: {i}, True: {j}")

correct= sum([1 for i, j in zip(y_class, test_y) if i==j])
print(f"Accuracy: {correct/len(test_y)*100:.2f}%")

