#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

using namespace std;

//*******************************************

// functors ===============================================

struct ThresholdToBinary {
  __device__
  int operator()(const int& value) const {
    return value > 0 ? 1 : 0;
  }
};


__device__ bool check_collinear(int x1 ,int y1 ,int x2 ,int y2 ,int x3 ,int y3)
{
  int a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
  if(a == 0) return 1;
  else return 0;

}

__device__ int distancecalc(int x1 ,int y1 ,int x2 ,int y2)
{
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}
// kernels===================================================



__global__
void tank_targets(int *tankdir ,int i ,int T , int* HP, int* dist,int *xcoord,int *ycoord)
{
    if(HP[threadIdx.x] > 0)
    {
      tankdir[threadIdx.x] = (threadIdx.x+i)%T;
      if(HP[tankdir[threadIdx.x]] > 0)
        dist[threadIdx.x] = distancecalc(xcoord[threadIdx.x],ycoord[threadIdx.x],xcoord[tankdir[threadIdx.x]],ycoord[tankdir[threadIdx.x]]);
      else dist[threadIdx.x] = INT_MAX;
    }

    else
    {
      tankdir[threadIdx.x] = -1;
      dist[threadIdx.x] = INT_MAX;
    }
  //printf(" %d,%d \n", threadIdx.x ,tankdir[threadIdx.x]);
}

__global__
void target_setting(int M,int N ,int* tankdir ,int *xcoord,int *ycoord,int* HP ,int* score, int T , int* dist , int* locking)
{
    __shared__ int lockit;
    bool flag;
    lockit = 1001;
    int old = atomicCAS(&lockit,threadIdx.x+1000,1);
    int tankid = blockIdx.x;
    int target = tankdir[blockIdx.x];
    int inlinetank = threadIdx.x;
    int finaltank = -1;
    int x1 = xcoord[tankid]  ,y1 = ycoord[tankid] ,x2 =xcoord[target] ,y2 =ycoord[target] ,x3 = xcoord[inlinetank] ,y3 = ycoord[inlinetank];
    int finaltarget_dist = INT_MAX;

    if(target != -1)
    {

       if(inlinetank != target&& tankid != inlinetank && check_collinear(x1 , y1 , x2 , y2 , x3 , y3) && HP[inlinetank]>0)
      {
            if ((x1 < x3 && x3 < x2) || (x2 < x3 && x3 < x1 || (y1 < y3 && y3 < y2) || (y2 < y3 && y3 < y1))) {

              finaltank = inlinetank;
              finaltarget_dist = distancecalc(x1,y1,x3,y3);


              //return "Third point is between the other two";
              //update tankdir

        }
        else
        {

        if (HP[target]<=0 &&((x1 < x2 && x2 < x3) || (x3 < x2 && x2 < x1) ||(y1 < y2 && y2 < y3) || (y3 < y2 && y2 < y1))) {


                finaltank = inlinetank;
                finaltarget_dist = distancecalc(x1,y1,x3,y3);


                //return "Second point is between the other two";
                //update tankdir



          }
        }
      }

    }
    __syncthreads();
    //atmoic area

     flag = true;
    do{
        old = atomicCAS(&lockit , threadIdx.x, (threadIdx.x+1)%T);
        if(old == threadIdx.x)
        {
          if(finaltarget_dist < dist[tankid] && finaltank != -1)
          {
            atomicExch(&tankdir[tankid],finaltank);
            atomicExch(&dist[tankid] , finaltarget_dist);
          }
          flag = false;



        }

    }while(flag);


    __syncthreads();
    if(tankid == inlinetank && target != -1){
        if(HP[tankdir[tankid]] <= 0 )
        atomicExch(&tankdir[tankid],-1);
    }

}

__global__
void updatescore(int *score,int* HP ,int* tankdir)
{
  int tank = threadIdx.x;
  int target = tankdir[threadIdx.x];

  if(target != -1)
  {
    atomicAdd(&score[tank],1);
    atomicSub(&HP[target],1);
  }

}

__global__
void testscore(int *t)
{
  printf("score : %d , %d \n" , threadIdx.x , t[threadIdx.x]);
}

__global__
void testhp(int *t)
{
  printf("hp : %d , %d \n" , threadIdx.x , t[threadIdx.x]);
}

__global__
void testprint(int *t)
{
  printf("%d , %d \n" , threadIdx.x , t[threadIdx.x]);
}


// Write down the kernels here


//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;


    FILE *inputfilepointer;

    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0;
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

// initial setup ===============================================================
  int *xcoord_d ,*ycoord_d, *score_d, *dist,*locking_H , *locking;

  thrust::device_vector<int> HP(T,H);
  thrust::device_vector<int> tankdir(T);
  thrust::device_vector<int> thrust_HP(T);
  thrust::device_vector<int> d_result(T);
  //thrust::device_vector<int> distance(T);

    // allocating cuda memory and copying it

  cudaMalloc(&xcoord_d , sizeof(int)*T);
  cudaMalloc(&ycoord_d , sizeof(int)*T);
  cudaMalloc(&score_d , sizeof(int)*T);
  cudaMalloc(&dist , sizeof(int)*T);
  cudaMalloc(&locking , sizeof(int));

  locking_H = (int*)malloc(sizeof(int));
  *locking_H = 0;

  cudaMemcpy(xcoord_d, xcoord, sizeof(int)*T, cudaMemcpyHostToDevice);
  cudaMemcpy(ycoord_d, ycoord, sizeof(int)*T, cudaMemcpyHostToDevice);
  cudaMemcpy(locking, locking_H, sizeof(int), cudaMemcpyHostToDevice);
    // converting to raw pointer to pass in kernel


  int* tankdir_ptr = thrust::raw_pointer_cast(tankdir.data());
  int* HP_ptr = thrust::raw_pointer_cast(HP.data());
  // setup for looping

  thrust::copy(HP.begin(), HP.end(), thrust_HP.begin());
  thrust::transform(thrust_HP.begin(), thrust_HP.end(), d_result.begin(), ThresholdToBinary());
      // Reduce to get the sum of transformed elements
  int sum = thrust::reduce(d_result.begin(), d_result.end(), 0, thrust::plus<int>());
  int i=0;

  // starting loops

    //testhp<<<1,T>>>(HP_ptr);
  while(sum >1 && ++i)
  {
      if(i % T == 0)
        continue;

      tank_targets<<<1,T>>>(tankdir_ptr,i,T,HP_ptr,dist,xcoord_d,ycoord_d);

      //testprint<<<1,T>>>(tankdir_ptr);

      target_setting<<<T,T , T*T*sizeof(int)>>>(M,N,tankdir_ptr,xcoord_d,ycoord_d,HP_ptr,score_d,T, dist , locking);

      //testprint<<<1,T>>>(tankdir_ptr);

      updatescore<<<1,T>>>(score_d, HP_ptr , tankdir_ptr );

      thrust::copy(HP.begin(), HP.end(), thrust_HP.begin());

      thrust::transform(thrust_HP.begin(), thrust_HP.end(), d_result.begin(), ThresholdToBinary());

  // Reduce to get the sum of transformed elements

      sum = thrust::reduce(d_result.begin(), d_result.end(), 0, thrust::plus<int>());

  //testscore<<<1,T>>>(score_d);
  //testhp<<<1,T>>>(HP_ptr);
  //testhp<<<1,T>>>(HP_ptr);
  //printf("%d\n",sum);

      cudaDeviceSynchronize();
  }

  cudaMemcpy(score, score_d, sizeof(int)*T, cudaMemcpyDeviceToHost);
  //for(int i =0 ; i<T ; i++)
  //{
    //  printf("%d\n",score[i]);
  //}


    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}