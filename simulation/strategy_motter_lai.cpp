#include <cmath>
#include <fstream>
#include <iostream>
#include <complex>
#include <algorithm>
#include <numeric>
#include <vector>
#include "RandomNumber.hpp"
#include "lib_file_io.cpp"
#include "graph.h"
#include "centrality.h"

using namespace std;
using namespace Snu::Cnrc;

vector<int> get_index_order_descend(const vector<double>& v_temp) {
    vector<pair<float, int> > v_sort(v_temp.size());

    for (int i = 0U; i < v_sort.size(); ++i) {
        v_sort[i] = make_pair(-v_temp[i], i);
    }

    random_device rd;
    pcg64 g(rd());

    shuffle(v_sort.begin(), v_sort.end(), g);
    sort(v_sort.begin(), v_sort.end());

    pair<double, int> rank;
    vector<int> result(v_temp.size());

    for(int i=0; i<v_sort.size(); i++){
        result[i] = v_sort[i].second;
    }
    return result;
}

int main(int argc, char *argv[]) {
    char* name_network = argv[1];
    double alpha = atof(argv[2]);
    char* name_strategy = argv[3];
    double p_immune = atof(argv[4]);

    RandomRealGenerator rnd(0,1);
    int error_loadtxt = 0;

    char name_network_alpha[400];
    sprintf(name_network_alpha, "%s_%g", name_network, alpha);

    char fname_network[400], fname_alpha[400], fname_bc[400], fname_score[400], fname_bc_one_removed[400], fnameout[400], fnameout_lcc[400];
    sprintf(fname_network, "network/%s.txt", name_network);
    sprintf(fname_bc, "bc/%s.txt", name_network);
    sprintf(fname_bc_one_removed, "bc_one_removed/%s.txt", name_network);
    sprintf(fnameout, "strategy/%s_%s_%g.txt", name_network_alpha, name_strategy, p_immune);
    sprintf(fnameout_lcc, "strategy_lcc/%s_%s_%g.txt", name_network_alpha, name_strategy, p_immune);

    Graph gr_star;
    Graph gr;

    gr_star.importFromListOfLinks(fname_network);
    int N = gr_star.getSize();

//    vector<int> conncomp(N, 0);
    vector<double> bc(N, 0);
    vector<double> bc_star(N, 0);
    vector<double> threshold(N, 0);
    vector<double> immune(N, 0);
    vector<vector<double>> bc_one_removed(N, vector<double>(N)); // bc when node n0 is removed: bc_one_removed[n0][:]

    vector<double> score(N, 0);
    if(strcmp(name_strategy, "degree") == 0){
        for(int n=0; n<N; n++){
            score[n] = gr_star.getNumFriendsOf(n);
        }
    }
    else if(strcmp(name_strategy, "random") == 0){
        for(int n=0; n<N; n++){
            score[n] = rnd();
        }
    }
    else if(strcmp(name_strategy, "avalanche_centrality") == 0){
        sprintf(fname_score, "avalanche_fraction/%s.txt", name_network_alpha);
        vector<double> temp1(N, 0);
        error_loadtxt += loadtxt(fname_score, temp1);

        sprintf(fname_score, "failure_fraction/%s.txt", name_network_alpha);
        vector<double> temp2(N, 0);
        error_loadtxt += loadtxt(fname_score, temp2);

        for(int n=0; n<N; n++){
            score[n] = temp1[n] * temp2[n];
        }
    }
    else if(strcmp(name_strategy, "avalanche_centrality_lcc") == 0){
        sprintf(fname_score, "avalanche_fraction_lcc/%s.txt", name_network_alpha);
        vector<double> temp1(N, 0);
        error_loadtxt += loadtxt(fname_score, temp1);

        sprintf(fname_score, "failure_fraction_lcc/%s.txt", name_network_alpha);
        vector<double> temp2(N, 0);
        error_loadtxt += loadtxt(fname_score, temp2);

        for(int n=0; n<N; n++){
            score[n] = temp1[n] * temp2[n];
        }
    }
    else if(strcmp(name_strategy, "bc") == 0 || strcmp(name_strategy, "evec") == 0){
        sprintf(fname_score, "%s/%s.txt", name_strategy, name_network);
        error_loadtxt += loadtxt(fname_score, score);
    }
    else{
        sprintf(fname_score, "%s/%s.txt", name_strategy, name_network_alpha);
        error_loadtxt += loadtxt(fname_score, score);
    }

    error_loadtxt += loadtxt(fname_bc, bc_star);
    error_loadtxt += loadtxt(fname_bc_one_removed, bc_one_removed);
    if(error_loadtxt){
        cout << "error in loadtxt" << endl;
        return 0;
    }

    for(int n=0; n<N; n++){
        threshold[n] = (1 + alpha) * bc_star[n];
    }

    vector<int> index_order = get_index_order_descend(score);

    vector<double> avalanche_fraction(N, 0);
    vector<double> failure_fraction(N, 0);
    vector<double> time(N, 0);

    int n_immune = round((double)p_immune/100 * N);
    for(int i_immune=0; i_immune<n_immune; i_immune++){
        int node_immune = index_order[i_immune];
        immune[node_immune] = 1;
    }

    for(int n0=0; n0<N; n0++){
        gr.copy(gr_star);
        gr.removeVertex(n0);
        avalanche_fraction[n0] += 1.0/N;
        failure_fraction[n0] += 1.0/N;

        int t = 0;
        while(true){
            if(t == 0){
                bc = bc_one_removed[n0];
            }
            else{
                getBCListOfGraph(gr, bc);
            }

            int is_stable = 1;
            for(int n=0; n<N; n++){
                if(immune[n] == 1){
                    continue;
                }

                if(bc[n] > threshold[n]){
                    gr.removeVertex(n);
                    avalanche_fraction[n0] += 1.0/N;
                    failure_fraction[n] += 1.0/N;
                    is_stable = 0;
                }
            }

            if(is_stable == 1){
                break;
            }
            t++;
        }

        time[n0] = t;


    }

    vector<double> output(1);
    double mean_failure_fraction = 0;
    for(int n=0; n<N; n++){
        mean_failure_fraction += failure_fraction[n]/N;
    }
    output[0] = mean_failure_fraction;
    savetxt(fnameout, output);

}
