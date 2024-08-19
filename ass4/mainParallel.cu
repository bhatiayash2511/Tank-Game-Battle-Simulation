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
        int slope1 = (y - y1) * (x2 - x1);
        int slope2 = (y2 - y1) * (x - x1);

        // Compare slopes to check if they are equal
        return slope1 == slope2;
    }
}

__global__ void tankInteractionKernel(Tanks d_tanks, int T, int currentRound, int* d_hpScoreCount){
    int tank_id = blockIdx.x;
    int tank_id_within = threadIdx.x;
    
    
    if (d_tanks.hp[tank_id] <= 0) {
        return; // Skip destroyed tanks
    }
    __syncthreads();
    if(blockIdx.x == threadIdx.x) return;
    __syncthreads();

    int target_id = (tank_id + currentRound) % T;  // Choose the next tank as the target
    int x = d_tanks.x[threadIdx.x];
    int y = d_tanks.y[threadIdx.x];
    __syncthreads();

    // Load tank coordinates and direction into shared memory
    __shared__ int x1, y1, x2, y2, dir;
    if (threadIdx.x == 0) {
        x1 = d_tanks.x[tank_id];
        y1 = d_tanks.y[tank_id];
        x2 = d_tanks.x[target_id];
        y2 = d_tanks.y[target_id];
        // Determine direction
        if (x1 <= x2 && y1 <= y2) dir = 1;
        else if (x1 > x2 && y1 <= y2) dir = 2;
        else if (x1 > x2 && y1 > y2) dir = 3;
        else if (x1 <= x2 && y1 > y2) dir = 4;
    }
    __syncthreads(); 


    
    
    bool result = isPointOnLine(x, y, x1, y1, x2, y2);
    if(result != true){
        return;
    }
    __syncthreads();

    __shared__ int it, temp, old, lockvar;
    if (threadIdx.x == 0) {
        it = -1;
        temp = 1e9;
        old = 10;
        lockvar = 0;
    }
    
    __syncthreads();  // Ensure all threads have initialized shared memory
    


    do {
    old = atomicCAS(&lockvar, 0,1);
        if(old == 0){
            if (dir == 1) {
                if (d_tanks.hp[tank_id_within] > 0 && x1 <= d_tanks.x[tank_id_within] && y1 <= d_tanks.y[tank_id_within]) {
                    if ((abs(x1 - d_tanks.x[tank_id_within]) + abs(y1 - d_tanks.y[tank_id_within])) < temp) {
                        temp = (abs(x1 - d_tanks.x[tank_id_within]) + abs(y1 - d_tanks.y[tank_id_within]));
                        it = tank_id_within;
                    }
                }
            }
            if (dir == 2) {
                if (d_tanks.hp[tank_id_within] > 0 && x1 > d_tanks.x[tank_id_within] && y1 <= d_tanks.y[tank_id_within]) {
                    if ((abs(x1 - d_tanks.x[tank_id_within]) + abs(y1 - d_tanks.y[tank_id_within])) < temp) {
                        temp = (abs(x1 - d_tanks.x[tank_id_within]) + abs(y1 - d_tanks.y[tank_id_within]));
                        it = tank_id_within;
                    }
                }
            }
            if (dir == 3) {
                if (d_tanks.hp[tank_id_within] > 0 && x1 > d_tanks.x[tank_id_within] && y1 > d_tanks.y[tank_id_within]) {
                    if ((abs(x1 - d_tanks.x[tank_id_within]) + abs(y1 - d_tanks.y[tank_id_within])) < temp) {
                        temp = (abs(x1 - d_tanks.x[tank_id_within]) + abs(y1 - d_tanks.y[tank_id_within]));
                        it = tank_id_within;
                    }
                }
            }
            if (dir == 4) {
                if (d_tanks.hp[tank_id_within] > 0 && x1 <= d_tanks.x[tank_id_within] && y1 > d_tanks.y[tank_id_within]) {
                    if ((abs(x1 - d_tanks.x[tank_id_within]) + abs(y1 - d_tanks.y[tank_id_within])) < temp) {
                        temp = (abs(x1 - d_tanks.x[tank_id_within]) + abs(y1 - d_tanks.y[tank_id_within]));
                        it = tank_id_within;
                    }
                }
            }
            lockvar = 0;
        }
    } while(old!=0);
    __syncthreads();
    if (it != -1) {
        
        atomicAdd(&d_hpScoreCount[it],1);
        atomicAdd(&d_hpScoreCount[T+tank_id],1);
    }
    __syncthreads();

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
    int *tanksLeftP = (int *)malloc(sizeof(int));
    *tanksLeftP = T;
    
    
    for (int i = 0; i < T; ++i) {
        tanks.x[i] = xcoord[i];
        tanks.y[i] = ycoord[i];
        tanks.score[i] = score[i];
        tanks.hp[i] = H;
    }

    Tanks d_tanks;
    int *d_tanksLeft;
    
    
    
    cudaMalloc((void **)&d_tanks.x, T * sizeof(int));
    cudaMalloc((void **)&d_tanks.y, T * sizeof(int));
    cudaMalloc((void **)&d_tanks.hp, T * sizeof(int));
    cudaMalloc((void **)&d_tanks.score, T * sizeof(int));
    cudaMalloc(&d_tanksLeft, sizeof(int));
    
    // Copy data from host to device
    cudaMemcpy(d_tanks.x, tanks.x, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tanks.y, tanks.y, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tanks.hp, tanks.hp, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tanks.score, tanks.score, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tanksLeft, tanksLeftP, sizeof(int), cudaMemcpyHostToDevice);
    

    int tanksLeft = T;
    int currentRound = 1;
    while (tanksLeft > 1) {  // Continue rounds until only one or zero tanks left
        if (currentRound % T == 0) {
            currentRound++;
            continue;
        }
        int *hpScoreCount = (int *)malloc(2 * T * sizeof(int));
        int *d_hpScoreCount;
        cudaMalloc(&d_hpScoreCount, 2 * T * sizeof(int));
        cudaMemcpy(d_hpScoreCount, hpScoreCount, 2*T*sizeof(int), cudaMemcpyHostToDevice);
        // vector<vector <int>> tankHitCount = f(T, tanks, currentRound); 
        tankInteractionKernel<<<T,T>>>(d_tanks, T, currentRound, d_hpScoreCount);
        cudaDeviceSynchronize();
        cudaMemcpy(hpScoreCount, d_hpScoreCount, 2*T*sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < T; ++i) {
            tanks.hp[i] -= hpScoreCount[i];
            tanks.score[i] += hpScoreCount[T+i];
            if (tanks.hp[i] <= 0 && hpScoreCount[i]) tanksLeft--;
        }
        cudaMemcpy(d_tanks.hp, tanks.hp, T * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tanks.score, tanks.score, T * sizeof(int), cudaMemcpyHostToDevice);
        currentRound++;
        cudaFree(d_hpScoreCount);

    }
    printf("checking code execuete properly or not");


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
