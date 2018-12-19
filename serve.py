from Simulation import Simulation, Node
from NetworkUCB import NetworkUCB
from util_functions import readMatFile

def getNodes(labels, attributes, classes):
    node_list = []
    for id, label in enumerate(labels):
        if labels[id] == 1:
            node = Node(id, True, classes[id][0] - 1, attributes[id])
        else:
            node = Node(id, False, classes[id][0] - 1, attributes[id])
        node_list.append(node)
    return node_list

def get_model_api(dataset):
    algorithms = {}

    # data_path : points to the location where dataset is stored
    data = readMatFile(dataset)
    labels = data["Label"]
    attributes = data["Attributes"]
    graph = data["Network"].toarray()
    classes = data["Class"]

    class_set = set([i[0] for i in classes])
    arm_num = len(class_set)

    all_nodes = getNodes(labels, attributes, classes)
    print ("arm_num = " + str(arm_num))

    context_dimension = len(attributes[0])
    print ("context_dimension = " + str(context_dimension))

    algorithms["NetworkUCB"] = NetworkUCB(context_dimension, arm_num, graph, all_nodes, ALPHA=0.2, LAMBDA=0.1, BETA=1.1, RHO=10)
    alg = algorithms["NetworkUCB"]

    iterations = 0
    training_iters = 0
    simExperiment = Simulation(iterations, algorithms, training_iters, all_nodes, arm_num, graph)

    return simExperiment, alg