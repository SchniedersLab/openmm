
#define PI (3.14159265359f)

/**
 * Initialize tree for execution, set Processed to 0, OKtoProcess=1 for leaves and out-of-bound,
 * reset self volume accumulators.
 */

__device__ void resetTreeCounters(unsigned const int padded_tree_size,
                                  unsigned const int tree_size,
                                  unsigned const int offset,
                                  int* __restrict__ ovProcessedFlag,
                                  int* __restrict__ ovOKtoProcessFlag,
                                  const int* __restrict__ ovChildrenStartIndex,
                                  const int* __restrict__ ovChildrenCount,
                                  int* __restrict__ ovChildrenReported) {
    const unsigned int id = threadIdx.x;  //the index of this thread in the workgroup
    const unsigned int nblock = blockDim.x; //size of work group
    unsigned int begin = offset + id;
    unsigned int size = offset + tree_size;
    unsigned int end = offset + padded_tree_size;

    for (int slot = begin; slot < end; slot += nblock) {
        ovProcessedFlag[slot] = (slot >= size) ? 1 : 0; //mark slots with overlaps as not processed
    }
    for (int slot = begin; slot < end; slot += nblock) {
        ovOKtoProcessFlag[slot] = (slot >= size) ? 0 : (ovChildrenCount[slot] == 0 ? 1
                                                                                   : 0); //marks leaf nodes (no children) as ok to process
    }
    for (int slot = begin; slot < end; slot += nblock) {
        ovChildrenReported[slot] = 0;
    }
}


//assume num. groups = num. tree sections
extern "C" __global__ void resetSelfVolumes(const int ntrees,
                                 const int* __restrict__ ovTreePointer,
                                 const int* __restrict__ ovAtomTreePointer,
                                 const int* __restrict__ ovAtomTreeSize,
                                 const int* __restrict__ ovAtomTreePaddedSize,
                                 const int* __restrict__ ovChildrenStartIndex,
                                 const int* __restrict__ ovChildrenCount,
                                 int* __restrict__ ovProcessedFlag,
                                 int* __restrict__ ovOKtoProcessFlag,
                                 int* __restrict__ ovChildrenReported,
                                 int* __restrict__ PanicButton) {
    unsigned int tree = blockIdx.x;      //initial tree
    if (PanicButton[0] > 0) return;
    while (tree < ntrees) {

        unsigned int offset = ovTreePointer[tree];
        unsigned int tree_size = ovAtomTreeSize[tree];
        unsigned int padded_tree_size = ovAtomTreePaddedSize[tree];
        resetTreeCounters(padded_tree_size, tree_size, offset,
                          ovProcessedFlag,
                          ovOKtoProcessFlag,
                          ovChildrenStartIndex,
                          ovChildrenCount,
                          ovChildrenReported);
        tree += gridDim.x;
    }
}


/**
 * Initialize tree for execution, set Processed to 0, OKtoProcess=1 for leaves and out-of-bound,
 * reset self volume accumulators.
 */
__device__ void resetTreeSection(unsigned const int padded_tree_size,
                                 unsigned const int offset,
                                 int* __restrict__ ovLevel,
                                 real* __restrict__ ovVolume,
                                 real* __restrict__ ovVsp,
                                 real* __restrict__ ovVSfp,
                                 real* __restrict__ ovSelfVolume,
                                 real* __restrict__ ovVolEnergy,
                                 int* __restrict__ ovLastAtom,
                                 int* __restrict__ ovRootIndex,
                                 int* __restrict__ ovChildrenStartIndex,
                                 int* __restrict__ ovChildrenCount,
                                 real4* __restrict__ ovDV1,
                                 real4* __restrict__ ovDV2,
                                 int* __restrict__ ovProcessedFlag,
                                 int* __restrict__ ovOKtoProcessFlag,
                                 int* __restrict__ ovChildrenReported) {
    const unsigned int nblock = blockDim.x; //size of thread block
    const unsigned int id = threadIdx.x;  //the index of this thread in the warp

    unsigned int begin = offset + id;
    unsigned int end = offset + padded_tree_size;

    for (int slot = begin; slot < end; slot += nblock) ovLevel[slot] = 0;
    for (int slot = begin; slot < end; slot += nblock) ovVsp[slot] = 1;
    for (int slot = begin; slot < end; slot += nblock) ovVSfp[slot] = 1;
    for (int slot = begin; slot < end; slot += nblock) ovSelfVolume[slot] = 0;
    for (int slot = begin; slot < end; slot += nblock) ovVolEnergy[slot] = 0;
    for (int slot = begin; slot < end; slot += nblock) ovLastAtom[slot] = -1;
    for (int slot = begin; slot < end; slot += nblock) ovRootIndex[slot] = -1;
    for (int slot = begin; slot < end; slot += nblock) ovChildrenStartIndex[slot] = -1;
    for (int slot = begin; slot < end; slot += nblock) ovChildrenCount[slot] = 0;
    //for(int slot=begin; slot<end ; slot+=nblock) ovDV1[slot] = (real4)0;
    //for(int slot=begin; slot<end ; slot+=nblock) ovDV2[slot] = (real4)0;
    for (int slot = begin; slot < end; slot += nblock) ovDV1[slot] = make_real4(0,0,0,0);
    for (int slot = begin; slot < end; slot += nblock) ovDV2[slot] = make_real4(0,0,0,0);
    for (int slot = begin; slot < end; slot += nblock) ovProcessedFlag[slot] = 0;
    for (int slot = begin; slot < end; slot += nblock) ovOKtoProcessFlag[slot] = 0;
    for (int slot = begin; slot < end; slot += nblock) ovChildrenReported[slot] = 0;
}

extern "C" __global__ void resetBuffer(unsigned const int bufferSize,
                            unsigned const int numBuffers,
                            real4* __restrict__ ovAtomBuffer,
                            real* __restrict__ selfVolumeBuffer,
                            long* __restrict__ selfVolumeBuffer_long,
                            long* __restrict__ gradBuffers_long) {

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    while (id < bufferSize) {
        selfVolumeBuffer_long[id] = 0;
        gradBuffers_long[id] = 0;
        gradBuffers_long[id + bufferSize] = 0;
        gradBuffers_long[id + 2 * bufferSize] = 0;
        gradBuffers_long[id + 3 * bufferSize] = 0;
        id += blockDim.x * gridDim.x;
    }

//TODOLater: Global memory fence needed or syncthreads sufficient?
    __syncthreads();
}


extern "C" __global__ void resetTree(const int ntrees,
                          const int* __restrict__ ovTreePointer,
                          const int* __restrict__ ovAtomTreePointer,
                          int* __restrict__ ovAtomTreeSize,
                          const int* __restrict__ ovAtomTreePaddedSize,
                          int* __restrict__ ovLevel,
                          real* __restrict__ ovVolume,
                          real* __restrict__ ovVsp,
                          real* __restrict__ ovVSfp,
                          real* __restrict__ ovSelfVolume,
                          real* __restrict__ ovVolEnergy,
                          int* __restrict__ ovLastAtom,
                          int* __restrict__ ovRootIndex,
                          int* __restrict__ ovChildrenStartIndex,
                          int* __restrict__ ovChildrenCount,
                          real4* __restrict__ ovDV1,
                          real4* __restrict__ ovDV2,
                          int* __restrict__ ovProcessedFlag,
                          int* __restrict__ ovOKtoProcessFlag,
                          int* __restrict__ ovChildrenReported,
                          int* __restrict__ ovAtomTreeLock,
                          int* __restrict__ NIterations) {
    unsigned int section = blockIdx.x; // initial assignment of warp to tree section
    while (section < ntrees) {
        unsigned int offset = ovTreePointer[section];
        unsigned int padded_tree_size = ovAtomTreePaddedSize[section];

        //each block resets one section of the tree
        resetTreeSection(padded_tree_size, offset,
                         ovLevel,
                         ovVolume,
                         ovVsp,
                         ovVSfp,
                         ovSelfVolume,
                         ovVolEnergy,
                         ovLastAtom,
                         ovRootIndex,
                         ovChildrenStartIndex,
                         ovChildrenCount,
                         ovDV1,
                         ovDV2,
                         ovProcessedFlag,
                         ovOKtoProcessFlag,
                         ovChildrenReported
        );
        if (threadIdx.x == 0) {
            ovAtomTreeLock[section] = 0;
            NIterations[section] = 0;
        }
        section += gridDim.x; //next section
    }
}
