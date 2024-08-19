// %%writefile ass4gpu.cu
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <vector>
using namespace std;

__device__ bool isPointOnLine(int xc, int yc, int x1, int y1, int x2, int y2) {
    
    // Check if the line is vertical
    if (x1 == x2) {
        return xc == x1;  // Tank lies on the line if x-coordinate matches
    }
    else if (y1 == y2) {
        return yc == y1;  // Tank lies on the line if y-coordinate matches
    }
    else {
        // Calculate slopes using type casting to ensure floating-point division
        int slope1 = (yc - y1) * (x2 - x1);
        int slope2 = (y2 - y1) * (xc - x1);

        // Compare slopes to check if they are equal
        return slope1 == slope2;
    }
}

__global__ void tankInteractionKernel(int *x,int *y,int *scoring,int *hp, int T,int currentRound,int* hd_hitScoreCount){
    int tank_id = blockIdx.x;
    int tank_id_within = threadIdx.x;
    
    int target_id = (tank_id + currentRound) % T;  // Choose the next tank as the target
    int xc = x[threadIdx.x];
    int yc = y[threadIdx.x];
    __syncthreads();

    // Load tank coordinates and direction into shared memory
    __shared__ int x1, y1, x2, y2, dir;
    if (threadIdx.x == 0) {
        x1 = x[tank_id];
        y1 = y[tank_id];
        x2 = x[target_id];
        y2 = y[target_id];
        // Determine direction
        if (x1 <= x2 && y1 <= y2) dir = 1;
        else if (x1 > x2 && y1 <= y2) dir = 2;
        else if (x1 > x2 && y1 > y2) dir = 3;
        else if (x1 <= x2 && y1 > y2) dir = 4;
    }
    __syncthreads(); 

    bool result = isPointOnLine(xc, yc, x1, y1, x2, y2);
    
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
            if (result && hp[tank_id] > 0 && tank_id != tank_id_within && dir == 1) {
                if (hp[tank_id_within] > 0 && x1 <= x[tank_id_within] && y1 <= y[tank_id_within]) {
                    if ((abs(x1 - x[tank_id_within]) + abs(y1 - y[tank_id_within])) < temp) {
                        temp = (abs(x1 - x[tank_id_within]) + abs(y1 - y[tank_id_within]));
                        it = tank_id_within;
                    }
                }
            }
            if (result && hp[tank_id] > 0 && tank_id != tank_id_within && dir == 2) {
                if (hp[tank_id_within] > 0 && x1 > x[tank_id_within] && y1 <= y[tank_id_within]) {
                    if ((abs(x1 - x[tank_id_within]) + abs(y1 - y[tank_id_within])) < temp) {
                        temp = (abs(x1 - x[tank_id_within]) + abs(y1 - y[tank_id_within]));
                        it = tank_id_within;
                    }
                }
            }
            if (result && hp[tank_id] > 0 && tank_id != tank_id_within && dir == 3) {
                if (hp[tank_id_within] > 0 && x1 > x[tank_id_within] && y1 > y[tank_id_within]) {
                    if ((abs(x1 - x[tank_id_within]) + abs(y1 - y[tank_id_within])) < temp) {
                        temp = (abs(x1 - x[tank_id_within]) + abs(y1 - y[tank_id_within]));
                        it = tank_id_within;
                    }
                }
            }
            if (result && hp[tank_id] > 0 && tank_id != tank_id_within && dir == 4) {
                if (hp[tank_id_within] > 0 && x1 <= x[tank_id_within] && y1 > y[tank_id_within]) {
                    if ((abs(x1 - x[tank_id_within]) + abs(y1 - y[tank_id_within])) < temp) {
                        temp = (abs(x1 - x[tank_id_within]) + abs(y1 - y[tank_id_within]));
                        it = tank_id_within;
                    }
                }
            }
            lockvar = 0;
        }
    } while(old!=0);
    __syncthreads();
    if (it != -1) {
        
        atomicAdd(&hd_hitScoreCount[it],1);
        atomicAdd(&hd_hitScoreCount[T+tank_id],1);
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

    
    int *x;   // X coordinate of each tank
    int *y;    // Y coordinate of each tank
    int *hp;   // Health Points of each tank
    int *scoring; // Score of each tank
    
    cudaHostAlloc(&x, T*sizeof(int), 0);
    cudaHostAlloc(&y, T*sizeof(int), 0);
    cudaHostAlloc(&hp, T*sizeof(int), 0);
    cudaHostAlloc(&scoring, T*sizeof(int), 0);
    
    
    
    for (int i = 0; i < T; ++i) {
        x[i] = xcoord[i];
        y[i] = ycoord[i];
        scoring[i] = score[i];
        hp[i] = H;
    }

    

    int tanksLeft = T;
    int currentRound = 1;
    while (tanksLeft > 1) {  // Continue rounds until only one or zero tanks left
        if (currentRound % T == 0) {
            currentRound++;
            continue;
        }
        int *hd_hitScoreCount;
        cudaHostAlloc(&hd_hitScoreCount, 2*T*sizeof(int), 0);
        
        tankInteractionKernel<<<T,T>>>(x,y,scoring,hp, T, currentRound, hd_hitScoreCount);
        cudaDeviceSynchronize();
        for(int i = 0; i < T; ++i) {
            printf("score %d , health %d \n", scoring[i], hp[i]);
            hp[i] -= hd_hitScoreCount[i];
            scoring[i] += hd_hitScoreCount[T+i];
            if (hp[i] <= 0 && hd_hitScoreCount[i]) tanksLeft--;
            
        }
        printf("score %d , health %d \n", scoring[T-1], hp[T-1]);

        currentRound++;
        cudaFreeHost(hd_hitScoreCount);

    }
    printf("checking code execuete properly or not");


    
    

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
        fprintf(outputfilepointer, "%d\n", scoring[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaFreeHost(x);
    cudaFreeHost(y);
    cudaFreeHost(hp);
    cudaFreeHost(scoring);
    cudaDeviceSynchronize();
    return 0;
}
