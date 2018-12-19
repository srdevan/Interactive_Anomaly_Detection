import os
import sys
import logging
from flask import Flask, request, jsonify, render_template, redirect, flash
from flask_cors import CORS
from serve import get_model_api
import json

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
CORS(app)

simExperiment = None
algorithm = None
picked_nodes = {}
anomalies = {}

@app.route("/api/v1/upload", methods=["POST"])
def upload():
    
    if "file" not in request.files:
        return "No file found!"

    file = request.files["file"]
    print(file)
    file.save(file.filename)
    global algorithm
    global simExperiment
    simExperiment, algorithm = get_model_api(file)

    return "File Uploaded successfully!"

@app.route("/api/v1/nodes/pick", methods=["GET"])
def pick_node():
    global picked_nodes
    picked_node = algorithm.decide(simExperiment.nodes)
    print(picked_node.contextFeatureVector)
    picked_nodes[picked_node.id] = picked_node

    picked_node_response = { "status": "success", "data1": { "node": picked_node.id, "attribute": str(picked_node.contextFeatureVector) } }
    return app.response_class(json.dumps(picked_node_response), content_type="application/json") 
    

@app.route("/api/v1/nodes/feedback/<node_id>", methods=["POST"])
def add_feedback(node_id):
    error = None

    optimal_reward = 1
    global picked_nodes

    if int(node_id) in picked_nodes.keys():
        reward = request.args.get("feedback")
        regret = optimal_reward - int(reward)
        
        global anomalies
        anomalies[int(node_id)] = regret

        algorithm.updateParameters(picked_nodes[int(node_id)], int(reward))
    else:
        error = "Node not picked yet"

    if error:
        temp = { "status": "failure", "errors": [error] }
        return app.response_class(json.dumps(temp), content_type="application/json")
    else:
        return "Feedback updated successfully"

@app.route("/api/v1/nodes/<node_id>/neighbors")
def fetch_neighbors(node_id):
    neighbor_feature_map = algorithm.getNeighborsFeatureVectorMap(picked_nodes[int(node_id)])
    
    return app.response_class(json.dumps(format_neighbor_feature_map(neighbor_feature_map)), content_type="application/json")

@app.route("/api/v1/anomalies")
def fetch_anomalies():
    global anomalies

    true_anomalies = []
    false_anomalies = []

    for key, value in anomalies.items():
        if value == 0:
            true_anomalies.append(key)
        else:
            false_anomalies.append(key)

    temp = {"data1":{"true_anomalies":true_anomalies, "false_anomalies":false_anomalies}}
    return app.response_class(json.dumps(temp), content_type="application/json")

@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

def format_neighbor_feature_map(neighbor_feature_map):
    data = []
    for neighbor, features in neighbor_feature_map.items():
        data.append(node_attributes(neighbor, features))
    
    return { "total_count": len(data), "data1": data, "errors": [] }

def node_attributes(node, features):
    return { "node": node, "attribute": features }

if __name__ == "__main__":
    # This is used when running locally.
    app.run(host='0.0.0.0', debug=True)