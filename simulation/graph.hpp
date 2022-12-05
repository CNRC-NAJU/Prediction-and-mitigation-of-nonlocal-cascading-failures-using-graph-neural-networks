#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_set>
#include <vector>

using std::cout;
using std::ifstream;
using std::ofstream;

using std::map;
using std::set;
using std::string;
using std::vector;

using std::distance;
using std::find;
using std::lower_bound;
using std::partial_sum;

using Node = unsigned int;

class Graph {
   public:
    Graph(const Node);
    void copy(Graph);

    void setNodesNum(const Node);
    void addNodesOf(const Node);

    void checkExistenceAndAddEdges(const Node, const Node);
    void addEdgeFromAtoB(const Node, const Node);
    void removeEdgeBfromA(const Node, const Node);
    void removeAllEdgesOf(const Node);
    void removeVertex(const Node);

    bool isLinked(const Node, const Node) const;
    bool isAllNodeConnected() const;

    void getDegreeDistribution();
    void printDegreeDistribution() const;

    void exportAsListOfLinks(string filename) const;
    void exportAsListOfLinks() const;
    void importFromListOfLinks(string filename);

    Node getSize() const;
    Node getNumFriendsOf(const Node) const;
    Node getNumEdges() const;

    const vector<Node>& getFriendsOf(const Node) const;
    const vector<vector<Node>>& getAdjacencyList() const;

   private:
    vector<vector<Node>> adjacencyList;
    unsigned int numEdges;
    map<Node, unsigned int> degreeDistribution;
};

Graph::Graph(const Node numVertices = 0) : adjacencyList(numVertices), numEdges(0) {
    adjacencyList.assign(numVertices, vector<Node>());
}

void Graph::copy(Graph motherGraph) {
    adjacencyList = motherGraph.adjacencyList;
    numEdges = motherGraph.numEdges;
}

void Graph::setNodesNum(const Node numNode) { adjacencyList.resize(numNode); }

void Graph::addNodesOf(const Node numNode) {
    adjacencyList.resize(adjacencyList.size() + numNode);
}

void Graph::checkExistenceAndAddEdges(const Node v1, const Node v2) {
    if (v1 != v2) {
        if (!isLinked(v1, v2)) {
            adjacencyList[v1].push_back(v2);
            adjacencyList[v2].push_back(v1);
            ++numEdges;
        }
    }
}

void Graph::addEdgeFromAtoB(const Node A, const Node B) {
    adjacencyList[B].push_back(A);
}

void Graph::removeEdgeBfromA(const Node A, const Node B) {
    adjacencyList[A].erase(find(adjacencyList[A].begin(), adjacencyList[A].end(), B));
}

void Graph::removeAllEdgesOf(const Node N) { adjacencyList[N].clear(); }

void Graph::removeVertex(const Node N) {
    for (int nb : adjacencyList[N]) {
        while (true) {
            auto it = find(adjacencyList[nb].begin(), adjacencyList[nb].end(), N);
            if (it == adjacencyList[nb].end()) {
                break;
            }
            adjacencyList[nb].erase(it, it + 1);
        }
    }
    adjacencyList[N].clear();
}

bool Graph::isLinked(const Node v1, const Node v2) const {
    if (v1 < v2)
        return (
            find(adjacencyList[v1].begin(), adjacencyList[v1].end(), v2) !=
            adjacencyList[v1].end()
        );

    else
        return (
            find(adjacencyList[v2].begin(), adjacencyList[v2].end(), v1) !=
            adjacencyList[v2].end()
        );
}

bool Graph::isAllNodeConnected() const {
    for (Node v = 0; v < adjacencyList.size(); ++v)
        if (adjacencyList[v].size() == 0) return false;

    return true;
}

void Graph::getDegreeDistribution() {
    for (auto friends : adjacencyList) {
        ++degreeDistribution[friends.size()];
    }
}

void Graph::printDegreeDistribution() const {
    for (auto degree : degreeDistribution)
        cout << degree.first << '\t' << degree.second << '\n';
}

void Graph::exportAsListOfLinks(string filename) const {
    ofstream fout(filename);
    if (fout.is_open()) {
        for (auto i = 0; i < adjacencyList.size(); ++i)
            for (auto friends : adjacencyList[i])
                if (friends > i) fout << i << '\t' << friends << '\n';
    }
    fout.close();
}

void Graph::exportAsListOfLinks() const {
    for (auto i = 0; i < adjacencyList.size(); ++i)
        for (auto friends : adjacencyList[i])
            if (friends > i) cout << i << '\t' << friends << '\n';
}

void Graph::importFromListOfLinks(string filename) {
    ifstream fin(filename);
    Node N1, N2;
    if (fin.is_open()) {
        while (fin >> N1 >> N2) {
            if (std::max(N1, N2) + 1 > getSize()) {
                setNodesNum(std::max(N1, N2) + 1);
            }
            checkExistenceAndAddEdges(N1, N2);
        }
    }

    fin.close();
}

Node Graph::getSize() const { return adjacencyList.size(); }

Node Graph::getNumFriendsOf(const Node v) const { return adjacencyList[v].size(); }

Node Graph::getNumEdges() const { return numEdges; }

const vector<Node>& Graph::getFriendsOf(const Node v) const { return adjacencyList[v]; }

const vector<vector<Node>>& Graph::getAdjacencyList() const { return adjacencyList; }
