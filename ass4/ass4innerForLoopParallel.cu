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

__device__ bool isPointOnLine(const int* x, const int* y, int pointIndex, int linePoint1Index, int linePoint2Index) {
    int xPoint = x[pointIndex], yPoint = y[pointIndex];
    int x1 = x[linePoint1Index], y1 = y[linePoint1Index];
    int x2 = x[linePoint2Index], y2 = y[linePoint2Index];

    // Check if the line is vertical
    if (x1 == x2) {
        return xPoint == x1;  // Tank lies on the line if x-coordinate matches
    }
    else if (y1 == y2) {
        return yPoint == y1;  // Tank lies on the line if y-coordinate matches
    }
    else {
        // Calculate slopes using type casting to ensure floating-point division
        double slope1 = static_cast<double>(yPoint - y1) / (xPoint - x1);
        double slope2 = static_cast<double>(y2 - y1) / (x2 - x1);

        // Compare slopes to check if they are equal
        return slope1 == slope2;
    }
}


__global__ void ffKernel(int T, int tank_id, int* x, int* y, int* hp, int* target_id, int x1, int y1, int dir, int* result) {
    int it = -1;
    int temp = 1e9;
    for (int i = 0; i < T; i++) {
        if (i == tank_id) {
            continue;
        }
        bool isOnLine = isPointOnLine(x, y, i, tank_id, *target_id);
        if (!isOnLine) {
            continue;
        }
        if (dir == 1 && hp[i] > 0 && x1 <= x[i] && y1 <= y[i]) {
            int distance = abs(x1 - x[i]) + abs(y1 - y[i]);
            if (distance < temp) {
                temp = distance;
                it = i;
            }
        }
        // Similar checks for other directions (2, 3, 4) omitted for brevity
    }
    result[blockIdx.x] = it;
}

int ff(int T, int tank_id, Tanks tanks, int target_id, int x1, int y1, int dir) {
    int* d_x, *d_y, *d_hp, *d_score, *d_result, *d_target_id;
    cudaMalloc((void**)&d_x, T * sizeof(int));
    cudaMalloc((void**)&d_y, T * sizeof(int));
    cudaMalloc((void**)&d_hp, T * sizeof(int));
    cudaMalloc((void**)&d_score, T * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));
    cudaMalloc((void**)&d_target_id, sizeof(int));

    cudaMemcpy(d_x, tanks.x, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, tanks.y, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hp, tanks.hp, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_score, tanks.score, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_id, &target_id, sizeof(int), cudaMemcpyHostToDevice);

    ffKernel<<<1, T>>>(T, tank_id, d_x, d_y, d_hp, d_target_id, x1, y1, dir, d_result);

    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_hp);
    cudaFree(d_score);
    cudaFree(d_result);
    cudaFree(d_target_id);

    return result;
}

vector<int> f(int T, Tanks tanks, int currentRound){
    vector<int> tankHitCount(T, 0);
    for (int tank_id = 0; tank_id < T; ++tank_id) {
        if (tanks.hp[tank_id] <= 0) {
            continue; // Skip destroyed tanks
        }

        int target_id = (tank_id + currentRound) % T;  // Choose the next tank as the target

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
        int it = ff(T, tank_id, tanks, target_id, x1, y1, dir);
        if (it != -1) {
            tankHitCount[it]++;
            tanks.score[tank_id]++;
        }
    }
    return tankHitCount;
}
int main(int argc, char **argv) {
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score;

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
    int tanksLeft = T;
    int currentRound = 1;
    while (tanksLeft > 1) {  // Continue rounds until only one or zero tanks left
        if (currentRound % T == 0) {
            currentRound++;
            continue;
        }
        
        vector<int> tankHitCount = f(T, tanks, currentRound); 
        for(int i = 0; i < T; ++i) {
            tanks.hp[i] -= tankHitCount[i];
            if (tanks.hp[i] <= 0 && tankHitCount[i]) tanksLeft--;
        }
        currentRound++;
    }

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++) {
        fprintf(outputfilepointer, "%d\n", tanks.score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    free(tanks.x);
    free(tanks.y);
    free(tanks.hp);
    free(tanks.score);
    cudaDeviceSynchronize();
    return 0;
}



