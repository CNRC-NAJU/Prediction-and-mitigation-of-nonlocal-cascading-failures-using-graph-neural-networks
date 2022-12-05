#include <time.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "RandomNumber.hpp"
#include "centrality.hpp"
#include "graph.hpp"
#include "lib_file_io.hpp"

using namespace std;
using namespace Snu::Cnrc;

/*
calculate the avalanche size of Motter-Lai model caused by removal of each node
Motter, Adilson E., and Ying-Cheng Lai. "Cascade-based attacks on complex networks."
Physical Review E 66.6 (2002): 065102. Lee, E. J., et al. "Robustness of the avalanche
dynamics in data-packet transport on scale-free networks." Physical Review E 71.5
(2005): 056108.
*/

int main(int argc, char* argv[]) {
    char* name_network = argv[1];
    double alpha = atof(argv[2]);

    // Define file names
    char fname_network[400], fnameout_avalanche_fraction[400], fname_bc[400],
        fname_bc_one_removed[400], fnameout_failure_fraction[400], fnameout_time[400];
    std::sprintf(fname_network, "data/edge_list/%s.txt", name_network);
    std::sprintf(fname_bc, "data/bc/%s.txt", name_network);
    std::sprintf(fname_bc_one_removed, "data/bc_one_removed/%s.txt", name_network);
    std::sprintf(
        fnameout_avalanche_fraction,
        "data/avalanche_fraction/%s_%g.txt",
        name_network,
        alpha
    );
    std::sprintf(
        fnameout_failure_fraction,
        "data/failure_fraction/%s_%g.txt",
        name_network,
        alpha
    );
    std::sprintf(fnameout_time, "data/time/%s_%g.txt", name_network, alpha);

    // Read graph
    Graph gr_star;
    Graph gr;
    gr_star.importFromListOfLinks(fname_network);
    int N = gr_star.getSize();

    // betweenness centrality (bc)
    vector<double> bc(N, 0);
    // bc when node n0 is removed: bc_one_removed[n0][:]
    vector<vector<double>> bc_one_removed(N, vector<double>(N));
    getBCListOfGraph(gr_star, bc);
    savetxt(fname_bc, bc);

    // Observables
    vector<double> time(N, 0);
    vector<double> avalanche_fraction(N, 0);
    vector<double> failure_fraction(N, 0);

    // Start Motter-Lai model simulation
    vector<double> threshold(N, 0);
    for (int n = 0; n < N; n++) {
        threshold[n] = (1 + alpha) * bc[n];
    }

    for (int n0 = 0; n0 < N; n0++) {
        gr.copy(gr_star);
        gr.removeVertex(n0);
        avalanche_fraction[n0] += 1.0 / N;
        failure_fraction[n0] += 1.0 / N;

        int t = 0;
        while (true) {
            // Calculate bc every step
            getBCListOfGraph(gr, bc);

            // Store bc when node n0 is removed
            if (t == 0) {
                bc_one_removed[n0] = bc;
            }

            // Scan all node if bc exceeds it's threshold
            int is_stable = 1;
            for (int n = 0; n < N; n++) {
                if (bc[n] <= threshold[n]) continue;

                gr.removeVertex(n);
                avalanche_fraction[n0] += 1.0 / N;
                failure_fraction[n] += 1.0 / N;
                is_stable = 0;
            }

            if (is_stable == 1) break;
            t++;
        }
        time[n0] = t;
    }

    savetxt(fname_bc_one_removed, bc_one_removed);
    savetxt(fnameout_time, time);
    savetxt(fnameout_avalanche_fraction, avalanche_fraction);
    savetxt(fnameout_failure_fraction, failure_fraction);
}
