#include <cmath>
#include <fstream>
#include <iostream>
#include <complex>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace std;

template <typename T>
void savetxt(char* fname, vector<T> matrix){
	int N = matrix.size();

	ofstream file(fname);
    for(int n=0; n<N; n++){
        file << matrix[n] << endl;
    }
    file.close();
}

template <typename T>
void savetxt(char* fname, vector<vector<T>> matrix){
	int N = matrix.size();

	ofstream file(fname);
    for(int n=0; n<N; n++){
        file << matrix[n][0];
        for(int m=1; m<matrix[n].size(); m++){
            file << "\t" << matrix[n][m];
        }
        file << endl;
    }
    file.close();
}

template <typename T>
int loadtxt(char* fname, vector<T>& matrix){
	int N = matrix.size();

	ifstream file(fname);
    if(file.fail()){
        return 1;
    }
    for(int n=0; n<N; n++){
        T elem;
        file >> elem;
        matrix[n] = elem;
	}
    file.close();
    return 0;
}

template <typename T>
int loadtxt(char* fname, vector<vector<T>>& matrix){
	int N = matrix.size();
	int M = matrix[0].size();

	ifstream file(fname);
    if(file.fail()){
        return 1;
    }
    for(int n=0; n<N; n++){
        for(int m=0; m<M; m++){
            T elem;
            file >> elem;
            matrix[n][m] = elem;
        }
	}
    file.close();
    return 0;
}

template <typename T>
void print_matrix(vector<T> matrix){
	int N = matrix.size();

    for(int n=0; n<N; n++){
        cout << matrix[n] << endl;
    }
}

template <typename T>
void print_matrix(vector<vector<T>> matrix){
	int N = matrix.size();

    for(int n=0; n<N; n++){
        cout << matrix[n][0];
        for(int m=1; m<matrix[n].size(); m++){
            cout << "\t" << matrix[n][m];
        }
        cout << endl;
    }
}