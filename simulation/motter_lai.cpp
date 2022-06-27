#include <cmath>
#include <fstream>
#include <iostream>
#include <complex>
#include <algorithm>
#include <numeric>
#include <vector>
#include <time.h>
#include "RandomNumber.hpp"
#include "lib_file_io.cpp"
#include "graph.h"
#include "centrality.h"

using namespace std;
using namespace Snu::Cnrc;

// calculate the avalanche size of Motter-Lai model caused by removal of each node
// Motter, Adilson E., and Ying-Cheng Lai. "Cascade-based attacks on complex networks." Physical Review E 66.6 (2002): 065102.
// Lee, E. J., et al. "Robustness of the avalanche dynamics in data-packet transport on scale-free networks." Physical Review E 71.5 (2005): 056108.

int main(int argc, char *argv[]) {
    char* name_network = argv[1];
    double alpha = atof(argv[2]);

    char fname_network[400], fnameout_avalanche_fraction[400], fnameout_failure_fraction[400], fnameout_avalanche_fraction_lcc[400], fnameout_failure_fraction_lcc[400], fnameout_time[400];
    sprintf(fname_network, "network/%s.txt", name_network);
    sprintf(fnameout_avalanche_fraction, "avalanche_fraction/%s_%g.txt", name_network, alpha);
    sprintf(fnameout_failure_fraction, "failure_fraction/%s_%g.txt", name_network, alpha);
    sprintf(fnameout_avalanche_fraction_lcc, "avalanche_fraction_lcc/%s_%g.txt", name_network, alpha);
    sprintf(fnameout_failure_fraction_lcc, "failure_fraction_lcc/%s_%g.txt", name_network, alpha);
    sprintf(fnameout_time, "time/%s_%g.txt", name_network, alpha);

    Graph gr_star;
    Graph gr;

    gr_star.importFromListOfLinks(fname_network);
    int N = gr_star.getSize();

    vector<double> threshold(N, 0);
    vector<double> avalanche_fraction(N, 0);
    vector<double> failure_fraction(N, 0);
    vector<double> avalanche_fraction_lcc(N, 0);
    vector<double> failure_fraction_lcc(N, 0);
    vector<double> time(N, 0);

    vector<int> conncomp(N, 0);

    vector<double> bc(N, 0);
    getBCListOfGraph(gr_star, bc);
    for(int n=0; n<N; n++){
        threshold[n] = (1 + alpha) * bc[n];
    }

    for(int n0=0; n0<N; n0++){
        gr.copy(gr_star);
        gr.removeVertex(n0);
        avalanche_fraction[n0] += 1.0/N;
        failure_fraction[n0] += 1.0/N;

        int t = 0;
        while(true){
            getBCListOfGraph(gr, bc);

            int is_stable = 1;
            for(int n=0; n<N; n++){
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

        int bin_lcc = getConnectedComponent(gr, conncomp);
        for(int n=0; n<N; n++){
            if(conncomp[n] != bin_lcc){
                avalanche_fraction_lcc[n0] += 1.0/N;
                failure_fraction_lcc[n] += 1.0/N;
            }
        }
    }

    savetxt(fnameout_avalanche_fraction, avalanche_fraction);
    savetxt(fnameout_failure_fraction, failure_fraction);
    savetxt(fnameout_avalanche_fraction_lcc, avalanche_fraction_lcc);
    savetxt(fnameout_failure_fraction_lcc, failure_fraction_lcc);
    savetxt(fnameout_time, time);
}
