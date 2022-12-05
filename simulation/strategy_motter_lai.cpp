#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "RandomNumber.hpp"
#include "centrality.hpp"
#include "graph.hpp"
#include "lib_file_io.hpp"

using namespace std;
using namespace Snu::Cnrc;

vector<int> get_index_order_descend(const vector<double>& v_temp) {
    vector<pair<float, int>> v_sort(v_temp.size());

    for (int i = 0U; i < v_sort.size(); ++i) {
        v_sort[i] = make_pair(-v_temp[i], i);
    }

    random_device rd;
    pcg64 g(rd());

    shuffle(v_sort.begin(), v_sort.end(), g);
    sort(v_sort.begin(), v_sort.end());

    pair<double, int> rank;
    vector<int> result(v_temp.size());

    for (int i = 0; i < v_sort.size(); i++) {
        result[i] = v_sort[i].second;
    }
    return result;
}

int main(int argc, char* argv[]) {
    char* name_network = argv[1];
    double alpha = atof(argv[2]);
    char* name_strategy = argv[3];
    double p_reinforce = atof(argv[4]);

    RandomRealGenerator rnd(0, 1);
    int error_loadtxt = 0;

    char name_network_alpha[400];
    std::sprintf(name_network_alpha, "%s_%g", name_network, alpha);

    char fname_network[400], fname_alpha[400], fname_bc[400], fname_score[400],
        fname_bc_one_removed[400], fnameout[400], fnameout_lcc[400];
    std::sprintf(fname_network, "data/edge_list/%s.txt", name_network);
    std::sprintf(fname_bc, "data/bc/%s.txt", name_network);
    std::sprintf(fname_bc_one_removed, "data/bc_one_removed/%s.txt", name_network);

    // Read graph
    Graph gr_star;
    Graph gr;
    gr_star.importFromListOfLinks(fname_network);
    int N = gr_star.getSize();

    // Read betweenness centrality (bc) & one removed
    vector<double> bc_star(N, 0);
    error_loadtxt += loadtxt(fname_bc, bc_star);

    vector<vector<double>> bc_one_removed(N, vector<double>(N));
    error_loadtxt += loadtxt(fname_bc_one_removed, bc_one_removed);

    // Load score for node to be reinforced
    vector<double> score(N, 0);
    const std::string name_strategy_str = std::string(name_strategy);
    if (name_strategy_str == "random") {
        for (int n = 0; n < N; n++) {
            score[n] = rnd();
        }
    } else if (name_strategy_str == "degree") {
        for (int n = 0; n < N; n++) {
            score[n] = gr_star.getNumFriendsOf(n);
        }
    } else if (name_strategy_str == "bc") {
        std::sprintf(fname_score, "data/bc/%s.txt", name_network);
        error_loadtxt += loadtxt(fname_score, score);
    } else if (name_strategy_str == "avalanche_fraction") {
        std::sprintf(fname_score, "data/avalanche_fraction/%s.txt", name_network_alpha);
        error_loadtxt += loadtxt(fname_score, score);
    } else if (name_strategy_str == "failure_fraction") {
        std::sprintf(fname_score, "data/failure_fraction/%s.txt", name_network_alpha);
        error_loadtxt += loadtxt(fname_score, score);
    } else if (name_strategy_str == "avalanche_centrality") {
        std::sprintf(fname_score, "data/avalanche_fraction/%s.txt", name_network_alpha);
        vector<double> temp1(N, 0);
        error_loadtxt += loadtxt(fname_score, temp1);

        std::sprintf(fname_score, "data/failure_fraction/%s.txt", name_network_alpha);
        vector<double> temp2(N, 0);
        error_loadtxt += loadtxt(fname_score, temp2);

        for (int n = 0; n < N; n++) {
            score[n] = temp1[n] * temp2[n];
        }
    } else if (name_strategy_str == "avalanche_centrality_gnn") {
        std::sprintf(
            fname_score, "data/avalanche_centrality_gnn/%s.txt", name_network_alpha
        );
        error_loadtxt += loadtxt(fname_score, score);
    } else {
        error_loadtxt += 1;
    }
    if (error_loadtxt) {
        cout << "error in loadtxt" << endl;
        return 0;
    }

    // Reinforce nodes
    vector<double> reinforced(N, 0);
    vector<int> index_order = get_index_order_descend(score);
    int n_reinforced = round(p_reinforce * N);
    for (int i_reinforced = 0; i_reinforced < n_reinforced; i_reinforced++) {
        int node_reinforced = index_order[i_reinforced];
        reinforced[node_reinforced] = 1;
    }

    // start Motter-Lai simulation
    vector<double> threshold(N, 0);
    for (int n = 0; n < N; n++) {
        threshold[n] = (1 + alpha) * bc_star[n];
    }
    vector<double> avalanche_fraction(N, 0);

    vector<double> bc(N, 0.0);
    for (int n0 = 0; n0 < N; n0++) {
        gr.copy(gr_star);
        gr.removeVertex(n0);
        avalanche_fraction[n0] += 1.0 / N;

        int t = 0;
        while (true) {
            if (t == 0) {
                bc = bc_one_removed[n0];
            } else {
                getBCListOfGraph(gr, bc);
            }

            int is_stable = 1;
            for (int n = 0; n < N; n++) {
                if (reinforced[n] == 1 || bc[n] <= threshold[n]) continue;

                gr.removeVertex(n);
                avalanche_fraction[n0] += 1.0 / N;
                is_stable = 0;
            }

            if (is_stable == 1) break;
            t++;
        }
    }

    // Store output: mean avalanche size
    vector<double> output(1);
    double mean_avalanche_fraction = 0.0;
    for (int n = 0; n < N; n++) {
        mean_avalanche_fraction += avalanche_fraction[n] / N;
    }
    output[0] = mean_avalanche_fraction;
    std::sprintf(
        fnameout, "data/mitigation/%s_%s_%g.txt", name_network_alpha, name_strategy, p_reinforce
    );
    savetxt(fnameout, output);
}
