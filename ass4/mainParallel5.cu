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









