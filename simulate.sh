#! /bin/bash

network_name=$1
a=$2
bin=simulation_N${network_name}A${a}

function build {
    g++ -O2 -std=c++17 -o ${bin} simulation/motter_lai.cpp
}

build
./${bin} ${network_name} ${a}
rm ${bin}
