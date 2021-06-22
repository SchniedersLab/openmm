#define PI (3.14159265359f)
/*
 * atomicAddLong is a functional alternative to atomicAdd in CUDA that can handle signed long long ints
 */
__device__ long long atomicAddLong(long long* address, long long val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, val +assumed);

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return old;
}

/**
 * Reduce the atom self volumes
 * energyBuffer could be linked to the selfVolumeBuffer, depending on the use
 */

extern "C" __global__ void reduceSelfVolumes_buffer(int num_atoms,
                                         int padded_num_atoms,
                                         int numBuffers,
                                         const int* __restrict__ ovAtomTreePointer,
                                         const real4* __restrict__ ovAtomBuffer,
				                         long long* __restrict__ gradBuffers_long,
				                         long long* __restrict__ selfVolumeBuffer_long,
				                         real* __restrict__ selfVolumeBuffer,
				                         real* __restrict__ selfVolume,
				                         const real* __restrict__ global_gaussian_volume, //atomic Gaussian volume
				                         const real* __restrict__ global_atomic_gamma, //atomic gammas
				                         real4* __restrict__ grad ){    //gradient wrt to atom positions and volume
  unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
  int totalSize = padded_num_atoms*numBuffers;
  real scale = 1/((real) 0x100000000);
  //accumulate self volumes
  unsigned int atom = id;
  while (atom < num_atoms) {
    // copy self volumes and gradients from long energy buffer to regular one
    if (global_atomic_gamma[atom] != 0.0) {
        grad[atom].x = scale*gradBuffers_long[atom];
        grad[atom].y = scale*gradBuffers_long[atom+padded_num_atoms];
        grad[atom].z = scale*gradBuffers_long[atom+2*padded_num_atoms];
    } else {
        grad[atom].x = 0.0;
        grad[atom].y = 0.0;
        grad[atom].z = 0.0;
    }
    selfVolume[atom] = scale*selfVolumeBuffer_long[atom];
    // divide gradient with respect to volume by volume of atom
    if(global_gaussian_volume[atom] > 0){
      grad[atom].w = scale*gradBuffers_long[atom+3*padded_num_atoms]/global_gaussian_volume[atom];
    }else{
      grad[atom].w = 0;
    }

   atom += blockDim.x*gridDim.x;
 }
  //TODOLater: Global memory fence needed or syncthreads sufficient?
 __syncthreads();
 
}


extern "C" __global__ void updateSelfVolumesForces(int update_energy,
				      int num_atoms,
				      int padded_num_atoms,
				      const int*  __restrict__ ovAtomTreePointer,
				      const real* __restrict__ ovVolEnergy,
				      const real4* __restrict__ grad, //gradient wrt to atom positions and volume
				      long long*   __restrict__ forceBuffers,
				      mixed*  __restrict__ energyBuffer){
  //update OpenMM's energies and forces
  unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int atom = id;
  while (atom < num_atoms) {
    // volume energy is stored at the 1-body level
    if(update_energy > 0){
      unsigned int slot = ovAtomTreePointer[atom];
      energyBuffer[id] += ovVolEnergy[slot];
      //alternative to the above, should give the same answer
      //energyBuffer[atom] += wen*global_atomic_gamma[atom]*selfVolume[atom];
    }
    atomicAddLong(&forceBuffers[atom                     ], ((-grad[atom].x)*0x100000000));
    atomicAddLong(&forceBuffers[atom +   padded_num_atoms], ((-grad[atom].y)*0x100000000));
    atomicAddLong(&forceBuffers[atom + 2*padded_num_atoms], ((-grad[atom].z)*0x100000000));
    atom += blockDim.x*gridDim.x;
 }
  //TODOLater: Global memory fence needed or syncthreads sufficient?
 __syncthreads();
 
}

