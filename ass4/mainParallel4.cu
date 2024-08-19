// %%writefile ass4gpu1.cu
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <vector>
using namespace std;
__device__ __host__ int calcDist(int x1, int y1, int x2, int y2){
    int dis = ((x2-x1)*(x2-x1)) + ((y2-y1)*(y2-y1));
    return dis; 
}
__device__ __host__ int calcDir(int x1, int y1, int x2, int y2){
    if (x1 <= x2 && y1 <= y2) return 1;
    else if (x1 > x2 && y1 <= y2) return 2;
    else if (x1 > x2 && y1 > y2) return 3;
    return 4;
}
__device__ __host__ bool isPointOnLine(int xc, int yc, int x1, int y1, int x2, int y2) {
    
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

__global__ void tankInteractionKernel(int *x,int *y,int *scoring,int *hp, int T,int currentRound,int* hd_ScoreCount,int* hd_HitCount){
    int target_id = (blockIdx.x + currentRound) % T;  // Choose the next tank as the target
    int xc = x[threadIdx.x];
    int yc = y[threadIdx.x];
    __syncthreads();

    // Load tank coordinates and direction into shared memory
    __shared__ int x1, y1, x2, y2, dir;
    if (threadIdx.x == 0) {
        x1 = x[blockIdx.x];
        y1 = y[blockIdx.x];
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
    printf("{%d,%d} & {%d,%d} line is %d collinear to {%d,%d}, dir is %d \n", x1,y1, x2,y2, result, xc,yc, dir);
    __syncthreads();

    __shared__ int it, smallest_dist, old, lockvar;
    if (threadIdx.x == 0) {
        it = -1;
        smallest_dist = 1e9;
        old = 10;
        lockvar = 0;
    }
    
    __syncthreads();  // Ensure all threads have initialized shared memory
    


    do {
    old = atomicCAS(&lockvar, 0,1);
        if(old == 0){
            
            if(hp[blockIdx.x] > 0 && hp[threadIdx.x] > 0){
                int distance = calcDist(x1, y1, xc, yc);
                int direction = calcDir(x1, y1, xc, yc);
                bool lieOnLine = isPointOnLine(xc, yc, x1, y1, x2, y2);
                if (threadIdx.x != blockIdx.x && lieOnLine && direction == dir && distance < smallest_dist)
                {
                    smallest_dist = distance;
                    it = threadIdx.x;
                }
                
            }



            lockvar = 0;
        }
    } while(old!=0);
    __syncthreads();
    if(threadIdx.x == 0){
        printf("%d -> %d and hit to %d\n", blockIdx.x, target_id, it);
        if (it != -1) {
        
          atomicAdd(&hd_HitCount[it],1);
          atomicAdd(&hd_ScoreCount[blockIdx.x],1);
        }
    }
    
    __syncthreads();

}



int main(int argc, char **argv) {
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score, *hp;

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
    int *health;   // Health Points of each tank
    int *scoring; // Score of each tank
    
    cudaMalloc(&x, T*sizeof(int));
    cudaMalloc(&y, T*sizeof(int));
    cudaMalloc(&health, T*sizeof(int));
    cudaMalloc(&scoring, T*sizeof(int));
    hp = (int *)malloc(T * sizeof(int));  
    for(int i = 0; i < T; i++){
        hp[i] = H;
    }
    cudaMemcpy(x, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(y, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(health, hp, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(scoring, score, T * sizeof(int), cudaMemcpyHostToDevice);
    
    
    
    

    

    int tanksLeft = T;
    int currentRound = 1;
    for(int i = 0; i < T; i++){
      printf("point (%d,%d) score %d Hp %d \n", xcoord[i], ycoord[i], score[i], hp[i]);
    }
    
    while (tanksLeft > 1) {  // Continue rounds until only one or zero tanks left
        if (currentRound % T == 0) {
            currentRound++;
            continue;
        }
        // Allocate memory on CPU
        int *ScoreCount = (int *)malloc(T * sizeof(int));
        int *HitCount = (int *)malloc(T * sizeof(int));
        int *hd_ScoreCount;
        int *hd_HitCount;
        cudaMalloc(&hd_ScoreCount, T*sizeof(int));
        cudaMalloc(&hd_HitCount, T*sizeof(int));

        cudaMemcpy(hd_ScoreCount, ScoreCount, T * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(hd_HitCount, HitCount, T * sizeof(int), cudaMemcpyHostToDevice);

        tankInteractionKernel<<<T,T>>>(x,y,scoring,health, T, currentRound, hd_ScoreCount, hd_HitCount);
        cudaDeviceSynchronize();
        
        cudaMemcpy(ScoreCount, hd_ScoreCount, T * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(HitCount, hd_HitCount, T * sizeof(int), cudaMemcpyDeviceToHost);

        for(int i = 0; i < T; ++i) {
            hp[i] -= HitCount[i];
            score[i] += ScoreCount[i];
            if (hp[i] <= 0) tanksLeft--;
        }
        for(int i = 0; i < T; i++){
            printf("point (%d,%d) score %d Hp %d \n", xcoord[i], ycoord[i], score[i], hp[i]);
        }
        printf("Round %d going to ended\n\n", currentRound);

        currentRound++;
        cudaFree(hd_ScoreCount);
        cudaFree(hd_HitCount);
        free(ScoreCount);
        free(HitCount);
        
        

    }
    printf("\nchecking code execuete properly or not\n");


    cudaMemcpy(hp, health, T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(score, scoring, T * sizeof(int), cudaMemcpyDeviceToHost);

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
        fprintf(outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    free(hp);
    
    cudaFree(x);
    cudaFree(y);
    cudaFree(health);
    cudaFree(scoring);
    cudaDeviceSynchronize();
    return 0;
}
