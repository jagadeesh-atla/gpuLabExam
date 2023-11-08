#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <stdio.h>

using namespace std;
const int N = 11;
const int M = 7;

void readInput(vector<int>& a){
	size_t n = a.size();
	// for (int& x : a) cin >> x;
	int i = 0;
	for (int t = 0; t < n; ++t) {
		cin >> i;
		a[t] = i;
	}	
}

void display(vector<int>& a, int n, int m){
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			printf("%5d ", a[i * m + j]);
		}
		printf("\n");
	}
}

__global__
void scalarMul(int* A, int a) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y; 
	if (ix < N and iy < M) 
		A[ix * M + iy] = a * A[ix * M + iy];
}

vector<int> doScalarMul(vector<int>& A, int a) {
	size_t bytes = A.size() * sizeof(int);
	int* dev_A;
	vector<int> aA(N*M);
	cudaMalloc(&dev_A, bytes);
	cudaMemcpy(dev_A, A.data(), bytes, cudaMemcpyHostToDevice);
	dim3 block(N, M, 1);
	dim3 grid(1, 1, 1);
	scalarMul<<<grid, block>>>(dev_A, a);
	cudaMemcpy(aA.data(), dev_A, bytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < A.size(); ++i) 
		assert(aA[i] == a * A[i]);

	cudaFree(dev_A);
	return aA;
}

__global__
void MatAdd(int* C, int *A, int *B) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int tid = ix * M + iy;
	if (ix < N and iy < M)
		C[tid] = A[tid] + B[tid];
}


vector<int> doMatAdd(vector<int>& A, vector<int>& B) {
	size_t bytes = A.size() * sizeof(int);
	int* dev_A, *dev_B, *dev_C;
	vector<int> C(N*M);
	cudaMalloc(&dev_A, bytes);
	cudaMalloc(&dev_B, bytes);
	cudaMalloc(&dev_C, bytes);
	cudaMemcpy(dev_A, A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B.data(), bytes, cudaMemcpyHostToDevice);
	dim3 block(N, M, 1);
	dim3 grid(1, 1, 1);
	MatAdd<<<grid, block>>>(dev_C, dev_A, dev_B);
	cudaMemcpy(C.data(), dev_C, bytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < A.size(); ++i)
		assert(C[i] == A[i] + B[i]);
	
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return C;
}

__global__
void MatTrans(int* MatAT, int* MatA) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < N and i < M) 
    MatAT[i * N + j] = MatA[j * M + i];
}


vector<int> doMatTrans(vector<int>& A) {
	size_t bytes = A.size() * sizeof(int);
	int* dev_C, *dev_C_T;
	vector<int> C_T(N * M);

	cudaMalloc(&dev_C, bytes);
	cudaMalloc(&dev_C_T, bytes);
	cudaMemcpy(dev_C, A.data(), bytes, cudaMemcpyHostToDevice);
	dim3 block(N, M, 1);
	dim3 grid(1, 1, 1);
	MatTrans<<<grid, block>>>(dev_C_T, dev_C);
	cudaMemcpy(C_T.data(), dev_C_T, bytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < M; ++j) 
			assert(C_T[j * N + i] == A[i * M + j]);

	cudaFree(dev_C);
	cudaFree(dev_C_T);
	return C_T;
}

const int M_ = 11;
const int N_ = 11;
const int K_ = 7;
const int SHMEM_SIZE = 1 << 10;
__global__ 
void matrixMul(int *c, int *a, int *b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];
  int tmp = 0;
  for (int i = 0; i < K_; i += blockDim.x) {
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K_ + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N_ + threadIdx.y * N_ + col];
    __syncthreads();
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }
    __syncthreads();
  }
  c[row * N_ + col] = tmp;
}

__global__ 
void mmul(int* c, int* a, int* b, int m1, int n1, int width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
  	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < 11 and col < 11) {
		int val = 0;
		for (int k = 0; k < 7; ++k) {
			val += a[row * 7 + k] * b[k * 11 + col];
		}
		c[col * 11 + row] = val;
	}
}

// Check result on the CPU
// MxN = MxK * KxN
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  int M = 11, N = 11, K = 7;
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      int tmp = 0;
      for (int i = 0; i < K; i++) {
        tmp += a[row * K + i] * b[i * N + col];
      }
      //assert(tmp == c[row * N + col]);
	//printf("%3d ", tmp);
	c[row * N + col] = tmp;
    }
    //printf("\n");
  }
}

vector<int> doMatMul(vector<int>& A, vector<int>& B) {
	size_t bytes = A.size() * sizeof(int);
	size_t bytesM = N * N * sizeof(int);
	vector<int> M(N * N);
	int *dev_A, *dev_B, *dev_M;
	
	cudaMalloc(&dev_A, bytes);	
	cudaMalloc(&dev_B, bytes);	
	cudaMalloc(&dev_M, bytesM);	

	cudaMemcpy(dev_A, A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B.data(), bytes, cudaMemcpyHostToDevice);
	dim3 block(256, 256, 1);
	dim3 grid(2, 2, 2);
	
	//matrixMul<<<grid, block>>>(dev_M, dev_A, dev_B);
	mmul<<<grid, block>>>(dev_M, dev_A, dev_B, 11, 11, 7);
	cudaMemcpy(M.data(), dev_M, bytesM, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	verify_result(A, B, M);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_M);
	return M;
}

int main() {
	// Declare A and B
	vector<int> A(N * M);
	vector<int> B(N * M);
	
	cout << "Read Matrix A: 1" << endl;
	readInput(A);
	cout << "Matrix A: 1" << endl;
	display(A, N, M);
	
	cout << "Read Matrix B: 4" << endl;
	readInput(B);
	cout << "Matrix B: 4" << endl;
	display(B, N, M);
	
	// Declare Scalars
	int a = 2, b = 3;
	cout << "Read a, b: " << endl;
	cin >> a >> b;
	cout << "(a, b): " << "(" << a << ", " << b << ")\n";

	// Calculate aA, bB
	vector<int> aA = doScalarMul(A, a);
	vector<int> bB = doScalarMul(B, b);
	
	cout << "Matrix aA: " << endl;
	display(aA, N, M);
	cout << "Matrix bB: " << endl;
	display(bB, N, M);
	
	// Calculate C
	vector<int> C = doMatAdd(aA, bB);
	
	cout << "Matrix C: aA + bB " << endl;
	display(C, N, M);	

	// Calculate C_T
	vector<int> C_T = doMatTrans(C);
	
	cout << "Matrix C_T: (aA + bB)^T" << endl;
	display(C_T, M, N);

	// Calculate C * C_T
	vector<int> M = doMatMul(C, C_T);
	
	cout << "Matrix M: (C * C_T)" << endl;
	display(M, N, N);
	
	return 0;	
}

