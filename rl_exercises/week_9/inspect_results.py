# import pickle
#
# with open("Results/dyna/seed_0.pkl","rb") as f:
#     data = pickle.load(f)
#
# print(data.keys())
#
# print(data["returns"][0])
# print(data["model_metrics"][0])
# print(data["multistep"][0])

import pickle

with open("Results/dyna2/seed_3.pkl", "rb") as f:
    data = pickle.load(f)

print(data["model_metrics"])
print(data["multistep"])
