// %%writefile GPUYashAss4.cu
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <vector>
using namespace std;
__device__ __host__ long long int calcDist(int x1, int y1, int x2, int y2){
    // long long int dis = ((x2-x1)*(x2-x1)) + ((y2-y1)*(y2-y1));
    // return dis; 
    long long int dis = (abs(x1 - x2) + abs(y1 - y2));
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
        long long int slope1 = (yc - y1) * (x2 - x1);
        long long int slope2 = (y2 - y1) * (xc - x1);

        // Compare slopes to check if they are equal
        return slope1 == slope2;
    }
}

__global__ void tankInteractionKernel(int *device_x,int *device_y,int *device_scoring,int *device_hp, int T,long long int currentRound, int *device_destroyed){
    int target_id = (blockIdx.x + currentRound) % T;  
    int xc = device_x[threadIdx.x];
    int yc = device_y[threadIdx.x];
    __syncthreads();

    
    __shared__ int x1, y1, x2, y2, dir;
    if (threadIdx.x == 0) {
        x1 = device_x[blockIdx.x];
        y1 = device_y[blockIdx.x];
        x2 = device_x[target_id];
        y2 = device_y[target_id];
        

        if (x1 <= x2 && y1 <= y2) dir = 1;
        else if (x1 > x2 && y1 <= y2) dir = 2;
        else if (x1 > x2 && y1 > y2) dir = 3;
        else if (x1 <= x2 && y1 > y2) dir = 4;
    }
    __syncthreads(); 

    bool result = isPointOnLine(xc, yc, x1, y1, x2, y2);
    //printf("{%d,%d} & {%d,%d} line is %d collinear to {%d,%d}, dir is %d \n", x1,y1, x2,y2, result, xc,yc, dir);
    //__syncthreads();

    __syncthreads(); 
    
    __shared__ long long int smallest_dist;
    
    __shared__ int it, old, lockvar;
    if (threadIdx.x == 0) {
        it = -1;
        smallest_dist = 1e12;
        old = 10;
        lockvar = 0;
    }
    
    __syncthreads();  
    long long int distance = calcDist(x1, y1, xc, yc);
    int direction = calcDir(x1, y1, xc, yc);
    bool lieOnLine = isPointOnLine(xc, yc, x1, y1, x2, y2);
    __syncthreads();
    if(!device_destroyed[blockIdx.x] && !device_destroyed[threadIdx.x]){
                
        if (threadIdx.x != blockIdx.x && lieOnLine && direction == dir)
        {
            do {
                old = atomicCAS(&lockvar, 0,1);
                if(old == 0){
        
                    // smallest_dist = distance;
                    // it = threadIdx.x;
                    if(distance < smallest_dist){
                        atomicMin((long long int*)&smallest_dist, distance);
                        atomicExch(&it, threadIdx.x);
                    }
                    atomicExch(&lockvar,0);
                }
            } while(old!=0);
        }
    }
    __syncthreads();


    if(threadIdx.x == 0){
        // printf("%d -> %d and hit to %d\n", blockIdx.x, target_id, it);
        if (it != -1) {
          atomicSub(&device_hp[it],1);
          atomicAdd(&device_scoring[blockIdx.x],1);
        }
    }
    
    __syncthreads();

}

// __global__
// void testvalues(int *device_hp,int *device_x,int *device_y,int *device_scoring)
// {
//   printf("TANK :  %d  = HP : %d SCORE : %d X : %d Y : %d \n",threadIdx.x,device_hp[threadIdx.x],device_scoring[threadIdx.x],device_x[threadIdx.x],device_y[threadIdx.x]);
// }

// __global__
// void values(int *device_hd_ScoreCount, int *device_hd_HitCount)
// {
//   printf("%d %d %d\n ",threadIdx.x,device_hd_ScoreCount[threadIdx.x],device_hd_HitCount[threadIdx.x]);
// }

__global__ void TankDestroyed(int *device_hp, int *device_tanksLeft, int *device_destroyed){
    if(device_hp[threadIdx.x] <= 0 && !device_destroyed[threadIdx.x]){
        atomicAdd(&device_destroyed[threadIdx.x], 1);
        atomicSub(device_tanksLeft, 1);
    }
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

    
    // printf("yoooooooooooooooooo");
    
    

    int *x = (int *)malloc(T * sizeof(int));   
    int *y = (int *)malloc(T * sizeof(int));    
    int *hp = (int *)malloc(T * sizeof(int));   
    int *scoring = (int *)malloc(T * sizeof(int));

    int *device_x;   
    int *device_y;    
    int *device_hp;   
    int *device_scoring;

    cudaMalloc(&device_x, T*sizeof(int));
    cudaMalloc(&device_y, T*sizeof(int));
    cudaMalloc(&device_hp, T*sizeof(int));
    cudaMalloc(&device_scoring, T*sizeof(int));
     
    
    
    for (int i = 0; i < T; ++i) {
        x[i] = xcoord[i];
        y[i] = ycoord[i];
        scoring[i] = 0;

        // Mine
        hp[i] = H;
    }

    cudaMemcpy(device_x, x, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, y, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_hp, hp, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_scoring, scoring, T * sizeof(int), cudaMemcpyHostToDevice);
    // printf("yoooooooooooooooooo\n");
    // testvalues<<<1,T>>>(device_hp,device_x,device_y,device_scoring);

    // printf("yoooooooooooooooooo");
    cudaDeviceSynchronize();
    int tanksLeft = T;
    long long int currentRound = 1;
    // for(int i = 0; i < T; i++){
    //   printf("point (%d,%d) scoring %d Hp %d \n", x[i], y[i], scoring[i], hp[i]);
    // }
    int *destroyed = (int *)malloc(T * sizeof(int));
    int *device_destroyed;
    cudaMalloc(&device_destroyed, T*sizeof(int));
    cudaMemcpy(destroyed, device_destroyed, T * sizeof(int), cudaMemcpyDeviceToHost);
    int *device_tanksLeft;
    cudaMalloc(&device_tanksLeft, sizeof(int));
    cudaMemcpy(device_tanksLeft, &tanksLeft, sizeof(int), cudaMemcpyHostToDevice);
    
    while (tanksLeft > 1) {  // Continue rounds until only one or zero tanks left
        if (currentRound % T == 0) {
            currentRound++;
            continue;
        }
        // printf("Round STarted\n");
        // printf("------------------------------------------------------- \n");
        // printf("------------------------------------------------------- \n");
        // printf("--------------------------------------------------------- \n");
        // printf("\n\n\n");
        tankInteractionKernel<<<T,T>>>(device_x,device_y,device_scoring,device_hp, T, currentRound, device_destroyed);
        cudaDeviceSynchronize();

        TankDestroyed<<<1,T>>>(device_hp, device_tanksLeft, device_destroyed);
        cudaDeviceSynchronize();
        cudaMemcpy(&tanksLeft,device_tanksLeft, sizeof(int), cudaMemcpyDeviceToHost);

        

        // cudaMemcpy(hp, device_hp, T * sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(scoring, device_scoring,  T * sizeof(int), cudaMemcpyDeviceToHost);
        // for(int i = 0; i < T; i++){
        //     printf("point (%d,%d) scoring %d Hp %d \n", x[i], y[i], scoring[i], hp[i]);
        // }
        // printf("Round %d going to end\n\n", currentRound);

        currentRound++;
        
        cudaDeviceSynchronize();

        

    }
    // printf("\nchecking code execuete properly or not\n");

    cudaMemcpy(hp, device_hp, T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(scoring, device_scoring,  T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    

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
    free(x);
    free(y);
    free(hp);
    free(scoring);
    free(destroyed);
    
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_hp);
    cudaFree(device_scoring);
    cudaFree(device_destroyed);
    cudaFree(device_tanksLeft);

    
    cudaDeviceSynchronize();
    return 0;
}
