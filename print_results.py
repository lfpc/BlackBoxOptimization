#Script must be executed inside the container

import pickle
with open("hall_of_fame.pkl", "rb") as f:
    hall_of_fame = pickle.load(f)
for individual_index in range(len(hall_of_fame)):
    individual=hall_of_fame[individual_index]
    print(f"Individual: {individual}, fitness value: {individual.fitness}, genes: {individual.genes}")