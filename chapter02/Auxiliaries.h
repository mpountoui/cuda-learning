#ifndef AUXILIARIES_H
#define AUXILIARIES_H

void allocateAndInitializeHostMemory(size_t nElem, float*& h_A, float*& hostRef, float*& gpuRef, size_t nBytes);
void allocateAndInitializeHostMemory(size_t nElem, float*& h_A, float*& h_B, float*& hostRef, float*& gpuRef, size_t nBytes);
void allocateDeviceMemory(size_t nBytes, float*& d_A, float*& d_C);
void allocateDeviceMemory(size_t nBytes, float*& d_A, float*& d_B, float*& d_C);
void copyInputsToDevice(float* h_A, float* h_B, float* d_A, float* d_B, size_t nBytes);
void freeMemory(float* h_A, float* h_B, float* hostRef, float* gpuRef, float* d_A, float* d_B, float* d_C);

#endif // AUXILIARIES_H
