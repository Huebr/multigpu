/*
 ============================================================================
 Name        : NewGpuRna.cu
 Author      : Pedro Jorge
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

/*
============================================================================
Name        : RNAFolding.cu

Author      : Pedro Jorge
Version     :
Copyright   : Your copyright notice
Description : CUDA compute reciprocals
============================================================================
*/

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//macros
#define MAX_FILENAME_SIZE 256
#define MAX_TEST_SIZE  40000
#define NBLOCK 16



//solve the RNA prediction problem
cudaError_t solverRNA(char *, int *, int);

#define cucheck_dev(call)                                   \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
	goto Error;                                         \
  }                                                         \
}


__host__ __device__ int hashing(int size,int i,int j){
  int deslocamento, deslocamento_externo;
  if(j<=0)return 0;
  const int k = 4;
  deslocamento = j - (k + i);
  if (deslocamento < 0)return 0;
  deslocamento_externo = (i * (2 * (size - k) - i + 1)) / 2;
  return (deslocamento + deslocamento_externo>=0) ? deslocamento + deslocamento_externo :0;
}
//reduce fast kernel

__inline__ __device__
int warpReduceSum(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val = max(val,__shfl_down(val, offset));
  return val;
}


__inline__ __device__
int blockReduceSum(int val) {

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

/*__device__ void deviceReduce(int *in, int* out, int N) {
  int sum = 0;
  //reduce multiple elements per thread
  for (int i = threadIdx.x;
       i < N;
       i += blockDim.x) {
    sum = max(sum,in[i]);
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
    *out=sum;
}
*/

__device__ bool canPair(int base1, int base2) {
  bool case1, case2;
	case1 = (base1 == 'C' && base2 == 'G') || (base1 == 'G' && base2 == 'C');
	case2 = (base1 == 'A' && base2 == 'U') || (base1 == 'U' && base2 == 'A');
	return (case1 || case2);
}

void printTime(int size,double time){
	FILE *ptr_file;
	char filename[MAX_FILENAME_SIZE];
	sprintf(filename, "output_time.txt");
	ptr_file = fopen(filename, "a");
	if(ptr_file==NULL){
		perror("Error opening time output file");
		exit(EXIT_FAILURE);
	}
	fprintf(ptr_file,"%d %f\n", size,time);
	fclose(ptr_file);
}

inline void printInfo(int *memo, const char* data, int size, int id) {
	FILE *fp;
	int i, j;
	char filename[MAX_FILENAME_SIZE];
	sprintf(filename, "output_info-%d.txt", id);
	fp = fopen(filename, "a");
	if (fp == NULL)
	{
		printf("Erro opening info file.");
		exit(1);
	}
	fprintf(fp, "--------------------new test---------------------\n");
	fprintf(fp, "Instance : %s\n", data);
	fprintf(fp, "Optimum value : %d\n", (memo[hashing(size,0,size-1)]));
	fprintf(fp, "Memoization Table : \n\n");
	for (i = 0; i < size; ++i) {
		for (j = 0; j < size; ++j) {
			fprintf(fp, "%d ", (memo[hashing(size,i,j)]));
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}
__device__ void __gpu_sync(int goalVal, volatile int *Arrayin, volatile int *Arrayout){
	int tid_in_blk = threadIdx.x*blockDim.y + threadIdx.y;
	int nBlockNum = gridDim.x*gridDim.y;
	int  bid = blockIdx.x*gridDim.y + blockIdx.y;

	if (tid_in_blk == 0){
		Arrayin[bid] = goalVal;
	}

	if (bid == 1){
		if (tid_in_blk < nBlockNum){
			while (Arrayin[tid_in_blk] != goalVal){
			}
		}
		__syncthreads();

		if (tid_in_blk < nBlockNum){
			Arrayout[tid_in_blk] = goalVal;
		}

	}

	if (tid_in_blk == 0){
		while (Arrayout[bid] != goalVal){

		}
	}
	__syncthreads();
}

__global__ void solverKernel(char *dev_data, int *dev_memo, int *dev_arrayIn, int *dev_arrayOut, int goalValue, int size)
{
	int i, j,t, opt;
  __shared__ char shared_data[MAX_TEST_SIZE];
  opt = 0;

  for(int temp = threadIdx.x;temp < size; temp+=blockDim.x){
    shared_data[temp]=dev_data[temp];
  }
  __syncthreads();

	for (int k = 5; k < size; k++){
		for(i=blockIdx.x;i<size - k;i+=gridDim.x){//seleciona subproblema
				j = i + k;
        opt = max(0,dev_memo[hashing(size,i,j - 1)]);
        for(t = i + threadIdx.x;t < j-4; t+=blockDim.x){//load estados
          if (canPair(shared_data[t], shared_data[j])) {
  					//if (t == 0) {
  						//opt = max(opt,1 + dev_memo[hashing(size,t+1,j - 1)]);
  					//}
  					//else {
  						opt = max(opt,1 + dev_memo[hashing(size,i,t - 1)] + dev_memo[hashing(size,t+1,j - 1)]);
  					//}
          }
      }
      __syncthreads();
      opt = blockReduceSum(opt);
      if(threadIdx.x==0)dev_memo[hashing(size,i,j)] = opt;
		}
		__gpu_sync(goalValue, dev_arrayIn, dev_arrayOut);
		goalValue++;
	}
}


bool canPairCPU(char base1, char base2) {
	bool case1, case2;
	case1 = (base1 == 'C' && base2 == 'G') || (base1 == 'G' && base2 == 'C');
	case2 = (base1 == 'A' && base2 == 'U') || (base1 == 'U' && base2 == 'A');
	return (case1 || case2);
}
void findSolution(FILE *fp, const char* data, int *memo, int size, int i, int j){
	if (i<j - 4){
		if (memo[hashing(size,i,j)] == memo[hashing(size,i,j - 1)]){
			findSolution(fp, data, memo, size, i, j - 1);
		}
		else{
			for (int t = i; t<j - 4; t++){
				if (canPairCPU(data[t], data[j])){
					if (t == 0){
						if ((memo[hashing(size,i,j)] - 1) == memo[hashing(size,t+1,j - 1)]){
							fprintf(fp, "%d %d undirected red\n", t, j);
							findSolution(fp, data, memo, size, t + 1, j - 1);
							break;
						}
					}
					else{
						if ((memo[i*size + j] - 1) == memo[hashing(size,i,t - 1)] + memo[hashing(size,t+1,j - 1)]){
							fprintf(fp, "%d %d undirected red\n", t, j);
							findSolution(fp, data, memo, size, t + 1, j - 1);
							findSolution(fp, data, memo, size, i, t - 1);
							break;
						}
					}
				}
			}
		}
	}
}
inline void createVertices(int id, const char* data, int size){
	FILE *fileptr;
	char filename[MAX_FILENAME_SIZE];
	sprintf(filename, "vertices-%d.csv", id);
	fileptr = fopen(filename, "a");
	if (fileptr == NULL){
		printf("error opening vertices file.");
	}
	else{
		fprintf(fileptr, "Id Label\n");
		for (int i = 0; i<size; i++){
			fprintf(fileptr, "%d %c\n", i, data[i]);
		}
	}
	fclose(fileptr);
}
int main(int argc,char *argv[])
{
	FILE *input;
	char *filename;
	char testRNA[MAX_TEST_SIZE];
	int result;
	int id;
	id = 0;
	cudaError_t cudaStatus;

	//Memory Allocation to file name

	//Reading filename
    filename = (char*)malloc(MAX_FILENAME_SIZE*sizeof(char));
    printf("Write name of input file : ");
    scanf("%s", filename);
	//Open File to read input test data
	input = fopen(filename, "r");

	//Testing input opening
	if (input == NULL) {
		printf("Error opening file, please try again.");
		return 1;
	}

	printf("\n\n---------------- Begin Tests --------------------\n\n");

	//Begin reading file and testing
	while (fscanf(input, "%s", testRNA) != EOF) {
		id++;
		//launch solverRNA
		cudaStatus = solverRNA(testRNA, &result, id);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "solverRNA failed!");
			return 1;
		}

		printf("%d : ", strlen(testRNA));
		printf("%d base pairs.\n", result);
	}

	printf("\n\n---------------- Ending Tests --------------------\n\n");



	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}

// Helper function for using CUDA to solve RNA prediction in parallel with objective function maximum number of bases
cudaError_t solverRNA(char *data, int *result, int id)
{

	char *dev_data = 0;//data in device
	int *dev_memo = 0;//memotable in device
	int *host_memo = 0;//memotable in host
	char *host_data = 0;
	int host_arrayIn[] = { 0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0};
	int host_arrayOut[] = { 0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0};
	int *dev_arrayIn;
	int *dev_arrayOut;
	float milliseconds;
	int goalValue = 1;
	int size = strlen(data);
	FILE *solution;
	char solutionName[MAX_FILENAME_SIZE];
	const int size_memo = (size*(size-3))/2;
	cudaError_t cudaStatus;


	//convert string to string
	host_data = data;
	// Choose which GPU to run on, change this on a multi-GPU system.
  checkCudaErrors(cudaSetDevice(0));
  checkCudaErrors(cudaDeviceEnablePeerAccess(1, 0));
  checkCudaErrors(cudaSetDevice(1));
  checkCudaErrors(cudaDeviceEnablePeerAccess(0, 0));

	// Allocate	CPU buffer to memoTable
	host_memo = (int *)calloc(size_memo, sizeof(int));

	// Allocate GPU buffer to memoTable
	cucheck_dev(cudaMalloc((void**)&dev_data, size*sizeof(char)));
	cucheck_dev(cudaMalloc((void**)&dev_memo, size_memo*sizeof(int)));
	cucheck_dev(cudaMalloc((void**)&dev_arrayIn, NBLOCK*sizeof(int)));
	cucheck_dev(cudaMalloc((void**)&dev_arrayOut, NBLOCK*sizeof(int)));

	// Copy input vectors from host memory to GPU buffers.
	cucheck_dev(cudaMemcpy(dev_memo, host_memo, size_memo * sizeof(int), cudaMemcpyHostToDevice));
	cucheck_dev(cudaMemcpy(dev_data, host_data, size * sizeof(char), cudaMemcpyHostToDevice));
	cucheck_dev(cudaMemcpy(dev_arrayIn, host_arrayIn, NBLOCK * sizeof(int), cudaMemcpyHostToDevice));
	cucheck_dev(cudaMemcpy(dev_arrayOut, host_arrayOut, NBLOCK * sizeof(int), cudaMemcpyHostToDevice));
	// Launch a kernel on the GPU with one thread for each element.
	cudaEventRecord(start);
	solverKernel <<< NBLOCK, 1024 >>> (dev_data, dev_memo, dev_arrayIn, dev_arrayOut, goalValue, size);
	// Check for any errors launching the kernel
	cucheck_dev(cudaGetLastError());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("solved in %f ms.\n", milliseconds);

	// Copy output vector from GPU buffer to host memory.
	cucheck_dev(cudaMemcpy(host_memo, dev_memo, size_memo*sizeof(int), cudaMemcpyDeviceToHost));

	*result = host_memo[hashing(size,0,size-1)];
  //printInfo(host_memo, data, size, id);
	printTime(size,milliseconds);
  /*
	createVertices(id, data, size);
	sprintf(solutionName, "edges-%d.csv", id);
	solution = fopen(solutionName, "a");
	if (solution == NULL){
		printf("error writing output connections.\n");
	}
	fprintf(solution, "Source Target Type Color\n");
	for (int i = 0; i<size - 1; i++){
		fprintf(solution, "%d %d undirected black\n", i, i + 1);
	}
	findSolution(solution, data, host_memo, size, 0, size - 1);
	fclose(solution);
  */
	Error:
		cudaFree(dev_memo);
		cudaFree(dev_data);
		cudaFree(dev_arrayIn);
		cudaFree(dev_arrayOut);
		free(host_memo);
		return cudaStatus;
}
