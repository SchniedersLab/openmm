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

//computes volume energy and self-volumes

//__global__ __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
extern "C" __global__ void computeSelfVolumes(const int ntrees,
                        const int *__restrict__ ovTreePointer,
                        const int *__restrict__ ovAtomTreePointer,
                        const int *__restrict__ ovAtomTreeSize,
                        int *__restrict__ NIterations,
                        const int *__restrict__ ovAtomTreePaddedSize,
                        const real *__restrict__ global_gaussian_exponent, //atomic Gaussian exponent
                        const int padded_num_atoms,
                        const int *__restrict__ ovLevel,
                        const real *__restrict__ ovVolume,
                        const real *__restrict__ ovVsp,
                        const real *__restrict__ ovVSfp,
                        const real *__restrict__ ovGamma1i,
                        const real4 *__restrict__ ovG,
                        real *__restrict__ ovSelfVolume,
                        real *__restrict__ ovVolEnergy,
                        const real4 *__restrict__ ovDV1,
                        real4 *__restrict__ ovDV2,
                        real4 *__restrict__ ovPF,
                        const int *__restrict__ ovLastAtom,
                        const int *__restrict__ ovRootIndex,
                        const int *__restrict__ ovChildrenStartIndex,
                        const int *__restrict__ ovChildrenCount,
                        int *__restrict__ ovProcessedFlag,
                        int *__restrict__ ovOKtoProcessFlag,
                        int *__restrict__ ovChildrenReported,
                        real4 *__restrict__ ovAtomBuffer,
                        long long *__restrict__ gradBuffers_long,
                        long long *__restrict__ selfVolumeBuffer_long,
                        real *__restrict__ selfVolumeBuffer) {
    const unsigned int id = threadIdx.x;
    const unsigned int gsize = blockDim.x;
    __shared__ volatile unsigned int nprocessed;
    __shared__ volatile unsigned int niterations;
    unsigned int tree = blockIdx.x;      //index of initial tree

    while (tree < ntrees) {
        unsigned int offset = ovTreePointer[tree]; //offset into tree
        unsigned int buffer_offset = tree * padded_num_atoms; // offset into buffer arrays

        unsigned int tree_size = ovAtomTreeSize[tree];
        unsigned int padded_tree_size = ovAtomTreePaddedSize[tree];
        unsigned int nsections = padded_tree_size / gsize;
        unsigned int ov_count = 0;

        // The tree for this atom is divided into sections each the size of a workgroup
        // The tree is walked bottom up one section at a time
        for (int isection = nsections - 1; isection >= 0; isection--) {
            unsigned int slot = offset + isection * gsize + id; //the slot to work on

            //reset accumulators
            ovVolEnergy[slot] = 0;
            ovSelfVolume[slot] = 0;
            ovDV2[slot] = make_real4(0, 0, 0, 0);
            int atom = ovLastAtom[slot];
            int level = ovLevel[slot];
            if (id == 0) niterations = 0;
//TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads();
            //
            // process section
            //
            do {
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
                if (id == 0) nprocessed = 0;
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
                int processed = ovProcessedFlag[slot];
                int ok2process = ovOKtoProcessFlag[slot];
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
                if (processed == 0 && ok2process == 0 && atom >= 0) {
                    if (ovChildrenReported[slot] == ovChildrenCount[slot]) {
                        ok2process = 1;
                        ovOKtoProcessFlag[slot] = 1;
                    }
                }
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();

                if (processed == 0 && ok2process > 0 && atom >= 0) {
                    //atomic_inc(&(nprocessed));
                    atomicAdd((unsigned int *) &nprocessed, 1);
                    real cf = level % 2 == 0 ? -1.0 : 1.0;
                    real volcoeff = level > 0 ? cf : 0;
                    real volcoeffp = level > 0 ? volcoeff / (float) level : 0;

                    //"own" volume contribution (volcoeff[level=0] for top root is automatically zero)
                    real self_volume = volcoeffp * ovVsp[slot] * ovVolume[slot];
                    double energy = ovGamma1i[slot] * self_volume;

                    //gather self volumes and derivatives from children

                    //dv1.xyz is (P)1..i in the paper
                    //dv1.w   is (F)1..i in the paper
                    //in relation to the gradient of the volume energy function
                    real4 dv1 = make_real4(0, 0, 0, volcoeffp * ovVSfp[slot] * ovGamma1i[slot]);
                    int start = ovChildrenStartIndex[slot];
                    int count = ovChildrenCount[slot];
                    if (count > 0 && start >= 0) {
                        for (int j = start; j < start + count; j++) {
                            if (ovLastAtom[j] >= 0) {
                                energy += ovVolEnergy[j];
                                self_volume += ovSelfVolume[j];
                                dv1 = make_real4(ovPF[j].x + dv1.x, ovPF[j].y + dv1.y, ovPF[j].z + dv1.z,
                                                 ovPF[j].w + dv1.w);
                            }
                        }
                    }

                    //stores new self_volume
                    ovSelfVolume[slot] = self_volume;

                    //stores energy
                    ovVolEnergy[slot] = energy;
                    //printf("ovVolEnergy: %u slot: %u \n", ovVolEnergy[slot], slot);
                    //printf("slot: %u\n", slot);

                    //
                    // Recursive rules for derivatives:
                    //
                    real an = global_gaussian_exponent[atom];
                    real a1i = ovG[slot].w;
                    real a1 = a1i - an;
                    real dvvc = dv1.w;//this is (F)1..i
                    ovDV2[slot] = make_real4(-ovDV1[slot].x * dvvc + (an / a1i) * dv1.x,
                                             -ovDV1[slot].y * dvvc + (an / a1i) * dv1.y,
                                             -ovDV1[slot].z * dvvc + (an / a1i) * dv1.z, //this gets accumulated later
                                             ovVolume[slot] * dvvc); //for derivative wrt volumei, gets divided by volumei later

                    ovPF[slot] = make_real4(ovDV1[slot].x * dvvc + (a1 / a1i) * dv1.x,
                                            ovDV1[slot].y * dvvc + (a1 / a1i) * dv1.y,
                                            ovDV1[slot].z * dvvc + (a1 / a1i) * dv1.z,
                                            ovDV1[slot].w * dvvc);

                    //mark parent ok to process counter
                    int parent_index = ovRootIndex[slot];
                    if (parent_index >= 0) {
                        atomicAdd(&ovChildrenReported[parent_index], 1);
                    }
                    ovProcessedFlag[slot] = 1; //mark as processed
                    ovOKtoProcessFlag[slot] = 0; //prevent more processing
                }
                if (id == 0) niterations += 1;
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
            } while (nprocessed > 0 && niterations < gsize); // loop until no more work is done
            //
            // End of processing for section
            //
            if (id == 0) {
                if (niterations > NIterations[tree]) NIterations[tree] = niterations;
            }

            // Updates energy and derivative buffer for this section
            //TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads();
//#ifdef SUPPORTS_64_BIT_ATOMICS
            if (atom >= 0) {
                real4 dv2 = ovDV2[slot];
                /*
                 * Commented Code in Original  Plugin
                atom_add(&forceBuffers[atom], (long) (dv2.x*0x100000000));
                atom_add(&forceBuffers[atom+padded_num_atoms], (long) (dv2.y*0x100000000));
                atom_add(&forceBuffers[atom+2*padded_num_atoms], (long) (dv2.z*0x100000000));
                atom_add(&gradVBuffer_long[atom], (long) (-dv2.w*0x100000000));
                */
                // if (ovSelfVolume[slot] != 0) {
                   atomicAddLong(&gradBuffers_long[atom], (dv2.x * 0x100000000));
                   atomicAddLong(&gradBuffers_long[atom + padded_num_atoms], (dv2.y * 0x100000000));
                   atomicAddLong(&gradBuffers_long[atom + 2 * padded_num_atoms], (dv2.z * 0x100000000));
                   atomicAddLong(&gradBuffers_long[atom + 3 * padded_num_atoms], (dv2.w * 0x100000000));
                   atomicAddLong( &selfVolumeBuffer_long[atom], (ovSelfVolume[slot] * 0x100000000));
                // }
                //printf("selfBuffer2: %d atom: %d\n", selfVolumeBuffer_long[atom], atom);
                // nothing to do here for the volume energy,
                // it is automatically stored in ovVolEnergy at the 1-body level
            }

//TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads();
        }

        // moves to next tree
        tree += gridDim.x;
//TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
    }
}

#ifdef NOTNOW
//same as self-volume kernel above but does not update self volumes
//TODO: __attribute__ ?
//__global__ __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
extern "C" __global__ void computeVolumeEnergy(const int ntrees,
  const int*  __restrict__ ovTreePointer,
  const int*  __restrict__ ovAtomTreePointer,
  const int*  __restrict__ ovAtomTreeSize,
        int*  __restrict__ NIterations,
  const int*  __restrict__ ovAtomTreePaddedSize,
  const real*  __restrict__ global_gaussian_exponent, //atomic Gaussian exponent
  const int*    __restrict__ ovLevel,
  const real*   __restrict__ ovVolume,
  const real*   __restrict__ ovVsp,
  const real*   __restrict__ ovVSfp,
  const real*   __restrict__ ovGamma1i,
  const real4*  __restrict__ ovG,
        real*   __restrict__ ovVolEnergy,
  const real4*  __restrict__ ovDV1,
        real4*  __restrict__ ovDV2,
        real4*  __restrict__ ovPF,
  const int*   __restrict__ ovLastAtom,
  const int*   __restrict__ ovRootIndex,
  const int*   __restrict__ ovChildrenStartIndex,
  const int*   __restrict__ ovChildrenCount,
        int*   __restrict__ ovProcessedFlag,
        int*   __restrict__ ovOKtoProcessFlag,
        int*   __restrict__ ovChildrenReported,
      real4*   __restrict__ ovAtomBuffer,
  long*    __restrict__ forceBuffers){
  const unsigned int id = threadIdx.x;
  const unsigned int gsize = blockDim.x;
  __local volatile unsigned int nprocessed;
  __local volatile unsigned int niterations;

  unsigned int tree = blockIdx.x;      //index of initial tree
  while(tree < ntrees){
    unsigned int offset = ovTreePointer[tree]; //offset into tree
    unsigned int buffer_offset = tree*PADDED_NUM_ATOMS; // offset into buffer arrays

    unsigned int tree_size = ovAtomTreeSize[tree];
    unsigned int padded_tree_size = ovAtomTreePaddedSize[tree];
    unsigned int nsections = padded_tree_size/gsize;
    unsigned int ov_count = 0;

    // The tree for this atom is divided into sections each the size of a workgroup
    // The tree is walked bottom up one section at a time
    for(int isection=nsections-1;isection >= 0; isection--){
      unsigned int slot = offset + isection*gsize + id; //the slot to work on

      //reset accumulators
      ovVolEnergy[slot] = 0;
      ovDV2[slot] = make_real4(0);
      int atom = ovLastAtom[slot];
      int level = ovLevel[slot];
      if(id == 0) niterations = 0;
      //TODOLater: Global memory fence needed or syncthreads sufficient?
      __syncthreads();
      //
      // process section
      //
      do{
    //TODOLater: Global memory fence needed or syncthreads sufficient?
    __syncthreads();
    if(id == 0) nprocessed = 0;
    //TODOLater: Global memory fence needed or syncthreads sufficient?
    __syncthreads();
    int processed = ovProcessedFlag[slot];
    int ok2process = ovOKtoProcessFlag[slot];
    //TODOLater: Global memory fence needed or syncthreads sufficient?
    __syncthreads();
    if(processed == 0 && ok2process == 0 && atom >= 0){
      if(ovChildrenReported[slot] == ovChildrenCount[slot]){
        ok2process = 1;
        ovOKtoProcessFlag[slot] = 1;
      }
    }
    //TODOLater: Global memory fence needed or syncthreads sufficient?
    __syncthreads();

    if(processed == 0 && ok2process > 0 && atom >= 0) {
      //atomic_inc(&(nprocessed));
      atomicInc(&(nprocessed));

      real cf = level % 2 == 0 ? -1.0 : 1.0;
      real volcoeff  = level > 0 ? cf : 0;
      real volcoeffp = level > 0 ? volcoeff/(float)level : 0;

      //"own" volume contribution (volcoeff[level=0] for top root is automatically zero)
      double energy = volcoeffp*ovGamma1i[slot]*ovVsp[slot]*ovVolume[slot];

      //gather self volumes and derivatives from children
      //dv.w is the gradient of the energy
      //real4 dv1 = (real4)(0,0,0,volcoeffp*ovVSfp[slot]*ovGamma1i[slot]);
      real4 dv1 = make_real4(0,0,0,volcoeffp*ovVSfp[slot]*ovGamma1i[slot]);
      int start = ovChildrenStartIndex[slot];
      int count = ovChildrenCount[slot];
      if(count > 0 && start >= 0){
        for(int j=start; j < start+count ; j++){
          if(ovLastAtom[j] >= 0 && ovLastAtom[j] < NUM_ATOMS_TREE){
        energy += ovVolEnergy[j];
        dv1 += ovPF[j];
          }
        }
      }

      //stores energy
      ovVolEnergy[slot] = energy;

      //
      // Recursive rules for derivatives:
      //
      real an = global_gaussian_exponent[atom];
      real a1i = ovG[slot].w;
      real a1 = a1i - an;
      real dvvc = dv1.w;
      //ovDV2[slot].xyz = -ovDV1[slot].xyz * dvvc  + (an/a1i)*dv1.xyz; //this gets accumulated later
      ovDV2[slot] = make_real4(-ovDV1[slot].x * dvcc + (an/a1i)*dv1.x,
                               -ovDV1[slot].y * dvcc + (an/a1i)*dv1.y,
                               -ovDV1[slot].z * dvcc + (an/a1i)*dv1.z,
                                ovDV2[slot].w);
      //ovPF[slot].xyz =  ovDV1[slot].xyz * dvvc  + (a1/a1i)*dv1.xyz;
      //ovPF[slot].w   =  ovDV1[slot].w   * dvvc;
      ovPF[slot] = make_real4(ovDV1[slot].x * dvvc  + (a1/a1i)*dv1.x,
                              ovDV1[slot].y * dvvc  + (a1/a1i)*dv1.y,
                              ovDV1[slot].z * dvvc  + (a1/a1i)*dv1.z,
                              ovDV1[slot].w * dvvc);

      //mark parent ok to process counter
      int parent_index = ovRootIndex[slot];
      if(parent_index >= 0){
        //atomic_inc(&(ovChildrenReported[parent_index]));
        atomicInc(&(ovChildrenReported[parent_index]));
      }
      ovProcessedFlag[slot] = 1; //mark as processed
      ovOKtoProcessFlag[slot] = 0; //prevent more processing
    }
    if(id==0) niterations += 1;
    //TODOLater: Global memory fence needed or syncthreads sufficient?
    __syncthreads();
      }while( nprocessed > 0 && niterations < gsize); // loop until no more work is done
      //
      // End of processing for section
      //
      if(id==0){
    if(niterations > NIterations[tree]) NIterations[tree] = niterations;
      }

      // Updates energy and derivative buffer for this section
      //TODOLater: Global memory fence needed or syncthreads sufficient?
      __syncthreads();
//#ifdef SUPPORTS_64_BIT_ATOMICS
      if(atom >= 0 && atom < NUM_ATOMS_TREE){
    real4 dv2 = -ovDV2[slot];
//	atom_add(&forceBuffers[atom], (long) (dv2.x*0x100000000));
//	atom_add(&forceBuffers[atom+PADDED_NUM_ATOMS], (long) (dv2.y*0x100000000));
//	atom_add(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (long) (dv2.z*0x100000000));
    atomicAdd(&forceBuffers[atom], (long) (dv2.x*0x100000000));
    atomicAdd(&forceBuffers[atom+PADDED_NUM_ATOMS], (long) (dv2.y*0x100000000));
    atomicAdd(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (long) (dv2.z*0x100000000));
    // nothing to do here for the volume energy,
    // it is automatically stored in ovVolEnergy at the 1-body level
      }
//#else
//      //
//      //without atomics can not accumulate in parallel due to "atom" collisions
//      //
//      if(id==0){
//	unsigned int tree_offset =  offset + isection*gsize;
//	for(unsigned int is = tree_offset ; is < tree_offset + gsize ; is++){ //loop over slots in section
//	  int at = ovLastAtom[is];
//	  if(at >= 0 && atom < NUM_ATOMS_TREE){
//	    // nothing to do here for the volume energy,
//	    // it is automatically stored in ovVolEnergy at the 1-body level
//	    //ovAtomBuffer[buffer_offset + at] += (real4)(ovDV2[is].xyz, 0); //.w element was used to store the energy
//	    ovAtomBuffer[buffer_offset + at] = make_real4(ovAtomBuffer[buffer_offset + at].x+ovDV2[is].x,
//	            ovAtomBuffer[buffer_offset + at].y + ovDV2[is].y,
//	            ovAtomBuffer[buffer_offset + at].z + ovDV2[is].z, 0);
//	  }
//	}
//      }
//#endif
      //TODOLater: Global memory fence needed or syncthreads sufficient?
      __syncthreads();
    }

    // moves to next tree
    tree += gridDim.x;
    //TODOLater: Global memory fence needed or syncthreads sufficient?
    __syncthreads();
  }
}
#endif
