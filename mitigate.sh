#! /bin/bash

network_name=$1
a=$2
strategy=$3
reinforce=$4
bin=simulation_N${network_name}A${a}S${strategy}R${reinforce}

function build {
    g++ -O2 -std=c++17 -o ${bin} simulation/strategy_motter_lai.cpp
}

build
./${bin} ${network_name} ${a} ${strategy} ${reinforce}
rm ${bin}
