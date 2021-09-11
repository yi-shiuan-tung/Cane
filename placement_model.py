import json
from collections import defaultdict
from sklearn.mixture import GaussianMixture
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# in meters
TRAY_WIDTH = 0.25
TRAY_HEIGHT = 0.37

NUT_HEIGHT = 0.089
NUT_WIDTH = 0.127
SCREW_HEIGHT = 0.092
SCREW_WIDTH = 0.115
LEG_HEIGHT = 0.0254
LEG_WIDTH = 0.1524
TOP_WIDTH = 0.0413
TOP_HEIGHT = 0.0413
FOOT_WIDTH = 0.0413
FOOT_HEIGHT = 0.0413

part_dimensions = {
    "top": [TOP_WIDTH, TOP_HEIGHT],
    "foot": [FOOT_WIDTH, FOOT_HEIGHT],
    "leg": [LEG_WIDTH, LEG_HEIGHT],
    "nut": [NUT_WIDTH, NUT_HEIGHT],
    "screw": [SCREW_WIDTH, SCREW_HEIGHT]
}


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
    parts = ["top", "leg", "foot", "screw", "nut"]

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
                dist_dict[obj1 + "-" + obj2].append([distance(pos1[0], pos2[0]), distance(pos1[1], pos2[1])])


def fit_gaussian(x):
    return GaussianMixture(n_components=1).fit(x)


def gurobi():
    relative_distances = get_relative_distances_from_json("prep_expert.json")
    gaussians = {}
    for pair, dists in relative_distances.items():
        gaussians[pair] = fit_gaussian(dists)

    # get a list of the unique parts
    parts = list(set(np.array(list(map(lambda x: x.split("-"), relative_distances.keys()))).flatten()))

    model = gp.Model("qp")

    # the variables are the x and y positions of each part
    variables = {}
    for part in parts:
        variables[part] = [model.addVar(lb=0., ub=TRAY_WIDTH, name=part+"-x"),
                           model.addVar(lb=0., ub=TRAY_HEIGHT, name=part+"-y")]

    # objective is to maximize probability density
    prob_density = 0
    for part1 in parts:
        for part2 in parts:
            if part1 != part2:
                dist = np.sqrt((variables[part1][0]-variables[part2][0])**2 + (variables[part1][1]-variables[part2][1])**2)
                prob_density += gaussians[part1 + "-" + part2].score(dist)

    model.setObjective(prob_density, GRB.MAXIMIZE)

    # add non-overlapping constraints
    for part1 in parts:
        for part2 in parts:
            if part1 != part2:
                if part1[-1].isnumeric():
                    part1 = part1[:-1]
                if part2[-1].isnumeric():
                    part2 = part2[:-1]

                # upper left point of part1
                ul_part1 = [variables[part1][0] - part_dimensions[part1][0]/2.0,
                            variables[part1][1] + part_dimensions[part1][1]/2.0]
                # bottom right point of part1
                br_part1 = [variables[part1][0] + part_dimensions[part1][0]/2.0,
                            variables[part1][1] - part_dimensions[part1][1]/2.0]

                # upper left point of part2
                ul_part2 = [variables[part2][0] - part_dimensions[part2][0]/2.0,
                            variables[part2][1] + part_dimensions[part2][1]/2.0]
                # bottom right point of part2
                br_part2 = [variables[part2][0] + part_dimensions[part2][0]/2.0,
                            variables[part2][1] - part_dimensions[part2][1]/2.0]

                model.addConstr(ul_part1[0] > br_part2[0] or ul_part2[0] > br_part1[0] or br_part1[1] > ul_part2[1]
                                or br_part2[1] > ul_part1[1], part1 + "-" + part2 + "-c")
    model.optimize()


def minimize_soft():
    relative_distances = get_relative_distances_from_json("prep_expert.json")
    gaussians = {}
    for pair, dists in relative_distances.items():
        gaussians[pair] = fit_gaussian(dists)

    # get a list of the unique parts
    parts = list(set(np.array(list(map(lambda x: x.split("-"), relative_distances.keys()))).flatten()))

    def objective_function(x, *args):
        """
        :param x: 1d array in the format [x1, y1, x2, y2 ... ]
        xn, yn are the x and y coordinates of the nth part
        """
        # objective is to maximize probability density
        prob_density = 0
        for i in range(len(parts)):
            for j in range(i, len(parts)):
                if i != j:
                    dist = [distance(x[i*2], x[j*2]), distance(x[i*2+1], x[j*2+1])]
                    prob_density += gaussians[parts[i] + "-" + parts[j]].score(np.array(dist).reshape((1, 2)))

        # add overlap costs
        overlaps = 0
        for i in range(len(parts)):
            for j in range(len(parts)):
                if i != j:
                    part1 = parts[i]
                    part2 = parts[j]
                    if part1[-1].isnumeric():
                        part1 = part1[:-1]
                    if part2[-1].isnumeric():
                        part2 = part2[:-1]

                    part1_pos = [x[i*2], x[i*2+1]]
                    part2_pos = [x[j*2], x[j*2+1]]
                    # upper left point of part1
                    ul_part1 = [part1_pos[0] - part_dimensions[part1][0]/2.0,
                                part1_pos[1] + part_dimensions[part1][1]/2.0]
                    # bottom right point of part1
                    br_part1 = [part1_pos[0] + part_dimensions[part1][0]/2.0,
                                part1_pos[1] - part_dimensions[part1][1]/2.0]

                    # upper left point of part2
                    ul_part2 = [part2_pos[0] - part_dimensions[part2][0]/2.0,
                                part2_pos[1] + part_dimensions[part2][1]/2.0]
                    # bottom right point of part2
                    br_part2 = [part2_pos[0] + part_dimensions[part2][0]/2.0,
                                part2_pos[1] - part_dimensions[part2][1]/2.0]
                    if not (ul_part1[0] >= br_part2[0] or ul_part2[0] >= br_part1[0] or br_part1[1] >= ul_part2[1] or
                            br_part2[1] >= ul_part1[1]):
                        overlaps += 1
        return -prob_density + overlaps

    bounds = []
    for i in range(len(parts)):
        part = parts[i]
        if part[-1].isnumeric():
            part = part[:-1]
        width, height = part_dimensions[part]
        # bounds for x-axis
        bounds.append([width/2.0, TRAY_WIDTH-width/2.0])
        # bounds for y-axis
        bounds.append([height/2.0, TRAY_HEIGHT-height/2.0])

    res = minimize(objective_function, np.ones(len(parts)*2)*0.1, bounds=bounds,
                   options={"disp": True, "maxiter": 1000})
    print(res.x)
    print(res.fun)
    print(res.message)

    part_positions = {}
    for i in range(len(parts)):
        part_positions[parts[i]] = [res.x[i*2], res.x[i*2+1]]
    draw_layout(part_positions)


def minimize_optimization():
    relative_distances = get_relative_distances_from_json("prep_expert.json")
    gaussians = {}
    for pair, dists in relative_distances.items():
        gaussians[pair] = fit_gaussian(dists)

    # get a list of the unique parts
    parts = list(set(np.array(list(map(lambda x: x.split("-"), relative_distances.keys()))).flatten()))

    def objective_function(x, *args):
        """
        :param x: 1d array in the format [x1, y1, x2, y2 ... ]
        xn, yn are the x and y coordinates of the nth part
        """
        # objective is to maximize probability density
        prob_density = 0
        for i in range(len(parts)):
            for j in range(i, len(parts)):
                if i != j:
                    dist = [distance(x[i*2], x[j*2]), distance(x[i*2+1], x[j*2+1])]
                    prob_density += gaussians[parts[i] + "-" + parts[j]].score(np.array(dist).reshape((1, 2)))
        return -prob_density

    # add non-overlapping constraints
    constraints = []
    for i in range(len(parts)):
        for j in range(len(parts)):
            if i != j:
                part1 = parts[i]
                part2 = parts[j]
                if part1[-1].isnumeric():
                    part1 = part1[:-1]
                if part2[-1].isnumeric():
                    part2 = part2[:-1]

                def constraint_function(x, *args):
                    part1_pos = [x[args[0]*2], x[args[0]*2+1]]
                    part2_pos = [x[args[1]*2], x[args[1]*2+1]]
                    # upper left point of part1
                    ul_part1 = [part1_pos[0] - part_dimensions[part1][0]/2.0,
                                part1_pos[1] + part_dimensions[part1][1]/2.0]
                    # bottom right point of part1
                    br_part1 = [part1_pos[0] + part_dimensions[part1][0]/2.0,
                                part1_pos[1] - part_dimensions[part1][1]/2.0]

                    # upper left point of part2
                    ul_part2 = [part2_pos[0] - part_dimensions[part2][0]/2.0,
                                part2_pos[1] + part_dimensions[part2][1]/2.0]
                    # bottom right point of part2
                    br_part2 = [part2_pos[0] + part_dimensions[part2][0]/2.0,
                                part2_pos[1] - part_dimensions[part2][1]/2.0]
                    return -1 + (ul_part1[0] >= br_part2[0]) + (ul_part2[0] >= br_part1[0]) + (br_part1[1] >= ul_part2[1]) \
                            + (br_part2[1] >= ul_part1[1])
                constraints.append({"type": "ineq", "fun": constraint_function, "args": [i, j]})

    bounds = []
    for i in range(len(parts)):
        part = parts[i]
        if part[-1].isnumeric():
            part = part[:-1]
        width, height = part_dimensions[part]
        # # bounds for x-axis
        # bounds.append([width/2.0, TRAY_WIDTH-width/2.0])
        # # bounds for y-axis
        # bounds.append([height/2.0, TRAY_HEIGHT-height/2.0])
        bounds.append([0, TRAY_WIDTH])
        bounds.append([0, TRAY_HEIGHT])

    res = minimize(objective_function, np.ones(len(parts)*2)*0.1, bounds=bounds, constraints=constraints,
                   options={"disp": True, "maxiter": 1000})
    print(res.x)
    print(res.fun)
    print(res.message)

    for c in constraints:
        for i in range(len(parts)):
            for j in range(len(parts)):
                if i != j:
                    r = c["fun"](res.x, i, j)
                    if r < 0:
                        print(parts[i], parts[j])

    part_positions = {}
    for i in range(len(parts)):
        part_positions[parts[i]] = [res.x[i*2], res.x[i*2+1]]
    draw_layout(part_positions)


def draw_layout(part_positions):
    """
    :param part_positions: Dict[String, Tuple[Float, Float]] Specifies the positions for each part
    """
    fig, ax = plt.subplots()

    # draw tray
    tray = plt.Rectangle((0, 0), TRAY_WIDTH, TRAY_HEIGHT, fill=False)
    ax.add_patch(tray)

    for part, pos in part_positions.items():
        if part[-1].isnumeric():
            part = part[:-1]
        width, height = part_dimensions[part]
        rect = plt.Rectangle((pos[0]-width/2.0, pos[1]-height/2.0), width, height, fill=False)
        ax.add_patch(rect)
        plt.text(pos[0], pos[1], part)
    plt.show()


if __name__ == "__main__":
    minimize_soft()
    # gurobi()
