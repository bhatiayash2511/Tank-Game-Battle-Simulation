#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <vector>
using namespace std;

struct Tanks {
    int *x;    // X coordinates of tanks
    int *y;    // Y coordinates of tanks
    int *hp;   // Health Points of tanks
    int *score; // Scores of tanks
};

// Function to check if a tank is on a line defined by two other tanks
__device__ bool isPointOnLine(int x, int y, int x1, int y1, int x2, int y2) {
    // Check if the line is vertical
    if (x1 == x2) {
        return x == x1;  // Tank lies on the line if x-coordinate matches
    }
    else if (y1 == y2) {
        return y == y1;  // Tank lies on the line if y-coordinate matches
    }
    else {
        // Calculate slopes using type casting to ensure floating-point division
        double slope1 = static_cast<double>(y - y1) / (x - x1);
        double slope2 = static_cast<double>(y2 - y1) / (x2 - x1);

        // Compare slopes to check if they are equal
        return slope1 == slope2;
    }
}

// CUDA kernel to handle tank interactions and health point deductions
__global__ void tankInteractionKernel(Tanks tanks, int T, int currentRound, int *tanksLeft) {
    int tank_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (tank_id < 5) printf("Adding this line for checking\n %d", tank_id);
    if (tank_id < T && tanks.hp[tank_id] > 0) {
        int target_id = (tank_id + currentRound) % T;

        int x1 = tanks.x[tank_id];
        int y1 = tanks.y[tank_id];
        int x2 = tanks.x[target_id];
        int y2 = tanks.y[target_id];

        int dir = 0;
        if (x1 <= x2 && y1 <= y2) {
            dir = 1;
        }
        if (x1 > x2 && y1 <= y2) {
            dir = 2;
        }
        if (x1 > x2 && y1 > y2) {
            dir = 3;
        }
        if (x1 <= x2 && y1 > y2) {
            dir = 4;
        }

        int it = -1;
        int temp = 1e9;
        for (int i = 0; i < T; i++) {
            if (i == tank_id) {
                continue;
            }

            if (dir == 1) {
                if (tanks.hp[i] > 0 && x1 <= tanks.x[i] && y1 <= tanks.y[i]) {
                    bool result = isPointOnLine(tanks.x[i], tanks.y[i], x1, y1, x2, y2);
                    if (result && (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i])) < temp) {
                        temp = (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i]));
                        it = i;
                    }
                }
            }
            else if (dir == 2) {
                if (tanks.hp[i] > 0 && x1 > tanks.x[i] && y1 <= tanks.y[i]) {
                    bool result = isPointOnLine(tanks.x[i], tanks.y[i], x1, y1, x2, y2);
                    if (result && (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i])) < temp) {
                        temp = (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i]));
                        it = i;
                    }
                }
            }
            else if (dir == 3) {
                if (tanks.hp[i] > 0 && x1 > tanks.x[i] && y1 > tanks.y[i]) {
                    bool result = isPointOnLine(tanks.x[i], tanks.y[i], x1, y1, x2, y2);
                    if (result && (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i])) < temp) {
                        temp = (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i]));
                        it = i;
                    }
                }
            }
            else if (dir == 4) {
                if (tanks.hp[i] > 0 && x1 <= tanks.x[i] && y1 > tanks.y[i]) {
                    bool result = isPointOnLine(tanks.x[i], tanks.y[i], x1, y1, x2, y2);
                    if (result && (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i])) < temp) {
                        temp = (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i]));
                        it = i;
                    }
                }
            }
        }
        if (it != -1) {
            atomicAdd(&tanks.score[tank_id], 1);
        }
    }
}

int main(int argc, char **argv) {
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score;
    int tanksLeft;

    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL) {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int *)malloc(T * sizeof(int));  // X coordinate of each tank
    ycoord = (int *)malloc(T * sizeof(int));  // Y coordinate of each tank
    score = (int *)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++) {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    Tanks tanks;
    tanks.x = (int *)malloc(T * sizeof(int));    // X coordinate of each tank
    tanks.y = (int *)malloc(T * sizeof(int));    // Y coordinate of each tank
    tanks.hp = (int *)malloc(T * sizeof(int));   // Health Points of each tank
    tanks.score = (int *)malloc(T * sizeof(int)); // Score of each tank

    for (int i = 0; i < T; ++i) {
        tanks.x[i] = xcoord[i];
        tanks.y[i] = ycoord[i];
        tanks.score[i] = score[i];
        tanks.hp[i] = H;
    }

    // Allocate memory on GPU
    Tanks d_tanks;
    cudaMalloc((void **)&d_tanks.x, T * sizeof(int));
    cudaMalloc((void **)&d_tanks.y, T * sizeof(int));
    cudaMalloc((void **)&d_tanks.hp, T * sizeof(int));
    cudaMalloc((void **)&d_tanks.score, T * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_tanks.x, tanks.x, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tanks.y, tanks.y, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tanks.hp, tanks.hp, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tanks.score, tanks.score, T * sizeof(int), cudaMemcpyHostToDevice);

    // Set up kernel launch parameters
    int blockSize = 256;
    int numBlocks = (T + blockSize - 1) / blockSize;

    // Perform tank interactions and health point deductions using CUDA kernel
    tankInteractionKernel<<<numBlocks, blockSize>>>(d_tanks, T, 1, &tanksLeft);

    // Copy the results back from the device to the host
    cudaMemcpy(tanks.x, d_tanks.x, T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tanks.y, d_tanks.y, T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tanks.hp, d_tanks.hp, T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tanks.score, d_tanks.score, T * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_tanks.x);
    cudaFree(d_tanks.y);
    cudaFree(d_tanks.hp);
    cudaFree(d_tanks.score);

    // Free memory on CPU
    free(tanks.x);
    free(tanks.y);
    free(tanks.hp);
    free(tanks.score);
    free(xcoord);
    free(ycoord);
    free(score);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    return 0;
}
