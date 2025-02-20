from pyclassify.classifier import kNN
from pyclassify.utils import  read_config
import argparse
import yaml


parser = argparse.ArgumentParser(description="Run kNN classification.")
parser.add_argument("--config", type=str, help="Path to config file.", required=True)

args = parser.parse_args()
config = read_config(args.config)
k = config['k']
dataset = config['dataset']

classification= kNN(k)



print(f"k: {k}, dataset: {dataset}")
X=[]
y=[]
with open(dataset, 'r') as f:
    for line in f:
        number_line=line.strip().split(',') 
        x=[float(i) for i in number_line[:-1]]
        X.append(x)
        y.append(1 if number_line[-1]=="g" else 0)

print(X)
N=len(y)
print(y)

i=int(N*0.2)
train_data=(X[:i], y[:i])
test_data=X[i:]
test_y=y[i:]

y_class=classification(train_data, test_data)
for i, j in zip(y_class, test_y):
    print(f"Predicted: {i}, True: {j}")

correct= sum([1 for i, j in zip(y_class, test_y) if i==j])
print(f"Accuracy: {correct/len(test_y)*100:.2f}%")

