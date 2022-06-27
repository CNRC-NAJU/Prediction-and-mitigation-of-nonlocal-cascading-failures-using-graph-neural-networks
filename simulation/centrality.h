#pragma once

#include <queue>
#include <stack>

#include "graph.h"

using std::stack;
using std::queue;
using std::priority_queue;


// Reference: U. Brandes, A Faster Algorithm for Betweenness Centrality, J. Math. Soc (2004)

void getBCListOfGraph(Graph& G, vector<double>& BC){

    Node N = G.getSize();  // Graph size

    queue<Node> Q; // initially empty
    stack<Node> S; // initially empty
    vector<Node> dist;  // distance from source
    vector<vector<Node> > Pred;  // list of predecessors on shortest paths from source
    vector<double> sigma;  // number of shortest paths fro source to v
    vector<double> delta;  // dependency of source on v

    // Initialized to 0
    BC.assign(N,0.0);

    for (unsigned s=0 ; s<N ; ++s) {
        // Single source shortest paths problem
            // Initialization
            Pred.assign(N,vector<Node>());
            dist.assign(N,-1);
            sigma.assign(N,0.0);

            dist[s]=0;
            sigma[s]=1;

            Q.push(s);
            while(!Q.empty()) {
                unsigned v = Q.front();
                Q.pop();
                S.push(v);

                for(Node w: G.getFriendsOf(v)){
                    // Path discovery // w found for the first time?
                    if (dist[w]== -1){
                        dist[w]=dist[v]+1;
                        Q.push(w);
                    }
                    // Path counting // edge(v,w) on a shortest path?
                    if (dist[w]==dist[v]+1){
                        sigma[w]=sigma[w]+sigma[v];
                        Pred[w].emplace_back(v);
                    }
                }
            }

        // Accumulation // back-propagation of dependencies
        delta.assign(N,0.0);
        while(!S.empty()) {
            Node w = S.top();
            S.pop();
            for(Node v : Pred[w])
                delta[v]+=sigma[v]/sigma[w]*(1.+delta[w]);
            if(w!=s)
                BC[w]+=delta[w];
        }
    }
}

// U. Brandes, On variants of shortest-path betweenness centrality and their generic computation, Social Networks (2008)

void getLoadListOfGraph(Graph& G, vector<double>& Load){
    Node N = G.getSize();  // Graph size

    queue<Node> Q; // initially empty
    stack<Node> S; // initially empty
    vector<Node> dist;  // distance from source
    vector<vector<Node> > Pred;  // list of predecessors on shortest paths from source
    vector<double> sigma;  // number of shortest paths fro source to v
    vector<double> delta;  // dependency of source on v

    // Initialized to 0
    Load.assign(N,0.0);

    for (Node s=0 ; s<N ; ++s) {
        // Single source shortest paths problem
            // Initialization
            Pred.assign(N,vector<Node>());
            dist.assign(N,-1);
            sigma.assign(N,0.0);

            dist[s]=0;
            sigma[s]=1;

            Q.push(s);
            while(!Q.empty()) {
                Node v = Q.front();
                Q.pop();
                S.push(v);

                for(Node w: G.getFriendsOf(v)){
                    // Path discovery // w found for the first time?
                    if (dist[w]== -1){
                        dist[w]=dist[v]+1;
                        Q.push(w);
                    }
                    // Path counting // edge(v,w) on a shortest path?
                    if (dist[w]==dist[v]+1){
                        sigma[w]=sigma[w]+sigma[v];
                        Pred[w].emplace_back(v);
                    }
                }
            }

        // Accumulation // back-propagation of dependencies
        delta.assign(N,1.0);
        while(!S.empty()){
            Node w = S.top();
            S.pop();
            for(Node v : Pred[w])
                delta[v]+=(delta[w]/Pred[w].size());

            Load[w]+=delta[w];
        }
    }
}

int getConnectedComponent(Graph& G, vector<int>& conncomp){
    //returns largest connected component index

    Node N = G.getSize();  // Graph size
    conncomp.assign(N, -1);
    vector<int> frontier(0);
    vector<int> frontier_new(0);

    int maxval_bin = -1;
    int maxdex_bin = -1;

    int i_bin = 0;
    for (Node s=0; s<N ;s++){
        if(conncomp[s] != -1){
            continue;
        }

        frontier.clear();
        frontier.push_back(s);

        int size_bin = 0;
        conncomp[s] = i_bin;
        size_bin++;

        while(!frontier.empty()){
            frontier_new.clear();
            for(Node v : frontier){
                for(Node w : G.getFriendsOf(v)){
                    if(conncomp[w] == -1){
                        frontier_new.push_back(w);
                        conncomp[w] = i_bin;
                        size_bin++;
                    }
                }
            }
            frontier = frontier_new;
        }

        if(size_bin > maxval_bin){
            maxval_bin = size_bin;
            maxdex_bin = i_bin;
        }

        i_bin++;
    }

    return maxdex_bin;
}