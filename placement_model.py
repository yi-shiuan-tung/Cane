import json
from collections import defaultdict
from sklearn.mixture import GaussianMixture
import numpy as np


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def get_relative_distances_from_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    distances = defaultdict(list)
    # iterate through each data point
    for sample in data["data"]:
        get_relative_distances_for_sample(distances, sample)

    return distances


def get_relative_distances_for_sample(dist_dict, sample_json):
    parts = ["top", "leg", "foot", "screw", "nut", "screwdriver", "seat"]

    # get the x, y positions of all objects
    positions = {}

    for part in parts:
        if part in ["top", "leg", "foot"]:
            for i in range(len(sample_json[part])):
                positions[part+str(i)] = sample_json[part][i]
        else:
            positions[part] = sample_json[part]

    # get pairwise relative distances for all objects
    for obj1, pos1 in positions.items():
        for obj2, pos2 in positions.items():
            if obj1 != obj2:
                dist_dict[obj1 + "-" + obj2 + "-x"].append(distance(pos1[0], pos2[0]))
                dist_dict[obj1 + "-" + obj2 + "-y"].append(distance(pos1[1], pos2[1]))


def fit_gaussian(x):
    return GaussianMixture(n_components=1).fit(x)


def main():
    relative_distances = get_relative_distances_from_json("prep_expert.json")
    gaussians = {}




if __name__ == "__main__":
    main()

