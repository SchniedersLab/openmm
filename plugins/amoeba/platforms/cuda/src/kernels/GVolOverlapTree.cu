#define PI (3.14159265359f)
#define Min_GVol (MIN_GVOL)
#define VolMinA (VOLMINA)
#define VolMinB (VOLMINB)

/* memory locking functions from http://www.cmsoft.com.br/opencl-tutorial/opencl-c99-atomics/
   by Douglas Coimbra de Andrade.
   occupied = 0 <- lock available
   occupied = 1 <- lock busy
  */
__device__ void GetSemaphor(int *semaphor) {
    //int occupied = atomic_xchg(semaphor, 1);
    int occupied = atomicExch(semaphor, 1);
    while (occupied > 0) //try until occupied = 0
    {
        //occupied = atomic_xchg(semaphor, 1);
        occupied = atomicExch(semaphor, 1);
    }
}

__device__ void ReleaseSemaphor(int *semaphor) {
    //int prevVal = atomic_xchg(semaphor, 0);
    int prevVal = atomicExch(semaphor, 0);
}

typedef struct {
    real4 posq;
    int ov_count;
    ATOM_PARAMETER_DATA
} AtomData;


typedef struct {
    int atom2;
    real gvol;
    real sfp;
    real gamma;
    real4 ovG;
    real4 ovDV1;
} OverlapData;


//this kernel initializes the tree with 1-body overlaps
//it assumes that no. of atoms in tree section is < groups size
extern "C" __global__ void InitOverlapTree_1body(
        unsigned const int num_padded_atoms,
        unsigned const int num_sections,
        unsigned const int reset_tree_size,
        const int* __restrict__ ovTreePointer,
        const int* __restrict__ ovNumAtomsInTree,
        const int* __restrict__ ovFirstAtom,
        int* __restrict__ ovAtomTreeSize,    //sizes of tree sections
        int* __restrict__ NIterations,
        const int* __restrict__ ovAtomTreePaddedSize,
        const int* __restrict__ ovAtomTreePointer,    //pointers to atoms in tree
        const real4* __restrict__ posq, //atomic positions
        const float* __restrict__ radiusParam, //atomic radius
        const float* __restrict__ gammaParam, //gamma
        const int* __restrict__ ishydrogenParam, //1=hydrogen atom
        real* __restrict__ GaussianExponent, //atomic Gaussian exponent
        real* __restrict__ GaussianVolume, //atomic Gaussian volume
        real* __restrict__ AtomicGamma, //atomic Gaussian gamma
        int* __restrict__ ovLevel, //this and below define tree
        real* __restrict__ ovVolume,
        real* __restrict__ ovVsp,
        real* __restrict__ ovVSfp,
        real* __restrict__ ovGamma1i,
        real4* __restrict__ ovG,
        real4* __restrict__ ovDV1,
        int* __restrict__ ovLastAtom,
        int* __restrict__ ovRootIndex,
        int* __restrict__ ovChildrenStartIndex,
        volatile int* __restrict__ ovChildrenCount) {
    const unsigned int id = threadIdx.x;
    unsigned int section = blockIdx.x;
    while (section < num_sections) {
        int natoms_in_section = ovNumAtomsInTree[section];
        int iat = id;
        while (iat < natoms_in_section) {
            int atom = ovFirstAtom[section] + iat;
            bool h = (ishydrogenParam[atom] > 0);
            real r = radiusParam[atom];
            real a = KFC / (r * r);
            real v = h ? 0 : 4.f * PI * pow(r, 3) / 3.f;
            real g = h ? 0 : gammaParam[atom];

            real4 c = posq[atom];
            GaussianExponent[atom] = a;
            GaussianVolume[atom] = v;
            AtomicGamma[atom] = g;

            int slot = ovAtomTreePointer[atom];
            ovLevel[slot] = 1;
            ovVolume[slot] = v;
            ovVsp[slot] = 1;
            ovVSfp[slot] = 1;
            ovGamma1i[slot] = g;
            //ovG[slot] = (real4)(c.xyz,a);
            //ovDV1[slot] = (real4)0.f;
            ovG[slot] = make_real4(c.x, c.y, c.z, a);
            ovDV1[slot] = make_real4(0, 0, 0, 0);
            ovLastAtom[slot] = atom;

            iat += blockDim.x;
        }
        if (id == 0) {
            if (reset_tree_size) ovAtomTreeSize[section] = natoms_in_section;
            NIterations[section] = 0;
        }

        section += gridDim.x;
        //TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
    }
}


//this kernel counts the no. of 2-body overlaps for each atom, stores in ovChildrenCount
extern "C" __global__ void InitOverlapTreeCount(
        const int* __restrict__ ovAtomTreePointer,    //pointers to atom trees
        const real4* __restrict__ posq, //atomic positions
        const real* __restrict__ global_gaussian_exponent, //atomic Gaussian exponent
        const real* __restrict__ global_gaussian_volume, //atomic Gaussian volume
#ifdef USE_CUTOFF
const int* __restrict__ tiles,
const unsigned int* __restrict__ interactionCount,
const int* __restrict__ interactingAtoms,
unsigned int maxTiles,
const ushort2* exclusionTiles,
#else
        unsigned int numTiles,
#endif
        int *__restrict__ ovChildrenCount
) {
    const unsigned int totalWarps = blockDim.x * gridDim.x / TILE_SIZE;
    const unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const unsigned int tgx = threadIdx.x & (TILE_SIZE - 1); //warp id in group
    const unsigned int tbx = threadIdx.x - tgx;           //id in warp
    __shared__ AtomData localData[FORCE_WORK_GROUP_SIZE];
    const unsigned int localAtomIndex = threadIdx.x;
    INIT_VARS

#ifdef USE_CUTOFF
    //OpenMM's neighbor list stores tiles with exclusions separately from other tiles

    // First loop: process tiles that contain exclusions
    // (this is imposed by OpenMM's neighbor list format, AGBNP does not actually have exclusions)
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
      const ushort2 tileIndices = exclusionTiles[pos];
      unsigned int x = tileIndices.x;
      unsigned int y = tileIndices.y;
      if(y>x) {unsigned int t = y; y = x; x = t;};//swap so that y<x

      unsigned int atom1 = y*TILE_SIZE + tgx;
      int parent_slot = ovAtomTreePointer[atom1];

      // Load atom data for this tile.
      real4 posq1 = posq[atom1];

      real a1 = global_gaussian_exponent[atom1];
      real v1 = global_gaussian_volume[atom1];

      unsigned int j = x*TILE_SIZE + tgx;
      localData[localAtomIndex].posq = posq[j];
      localData[localAtomIndex].g.w = global_gaussian_exponent[j];
      localData[localAtomIndex].v = global_gaussian_volume[j];

      SYNC_WARPS;
      if(y==x){//diagonal tile

        unsigned int tj = tgx;
        for (j = 0; j < TILE_SIZE; j++) {

      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real4 delta = make_real4(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z, 0);

      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      real a2 = localData[localAtom2Index].g.w;
      real v2 = localData[localAtom2Index].v;
      int atom2 = x*TILE_SIZE+tj;

      if (atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && atom1 < atom2 && r2 < CUTOFF_SQUARED) {
        COMPUTE_INTERACTION_COUNT
      }
      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
        }

      }else{//off-diagonal tile, pairs are unique, don't need to check atom1<atom2

        unsigned int tj = tgx;
        for (j = 0; j < TILE_SIZE; j++) {

      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real4 delta = make_real4(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z, 0);

      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      real a2 = localData[localAtom2Index].g.w;
      real v2 = localData[localAtom2Index].v;
      int atom2 = x*TILE_SIZE+tj;

      if (atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && r2 < CUTOFF_SQUARED) {
        COMPUTE_INTERACTION_COUNT
      }
      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
        }

      }
      SYNC_WARPS;
    }
#endif //USE_CUTOFF

    //second loop, tiles without exclusions or all interactions if not using cutoffs
#ifdef USE_CUTOFF
    __shared__ int atomIndices[FORCE_WORK_GROUP_SIZE];
    unsigned int numTiles = interactionCount[0];
    if(numTiles > maxTiles)
      return; // There wasn't enough memory for the neighbor list.
#endif
    int pos = (int) (warp * (long) numTiles / totalWarps);
    int end = (int) ((warp + 1) * (long) numTiles / totalWarps);
    while (pos < end) {
#ifdef USE_CUTOFF
        // y-atom block of the tile
        // atoms in x-atom block (y <= x) are retrieved from interactingAtoms[] below
        unsigned int y = tiles[pos];
        //unsigned int iat = y*TILE_SIZE + tgx;
        //unsigned int jat = interactingAtoms[pos*TILE_SIZE + tgx];
#else
        // find x and y coordinates of the tile such that y <= x
        int y = (int) floor(NUM_BLOCKS + 0.5f - SQRT((NUM_BLOCKS + 0.5f) * (NUM_BLOCKS + 0.5f) - 2 * pos));
        int x = (pos - y * NUM_BLOCKS + y * (y + 1) / 2);
        if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
            y += (x < y ? -1 : 1);
            x = (pos - y * NUM_BLOCKS + y * (y + 1) / 2);
        }
#endif

        unsigned int atom1 = y * TILE_SIZE + tgx;

        // Load atom data for this tile.
        int tree_pointer1 = ovAtomTreePointer[atom1];
#ifndef USE_CUTOFF
        //the parent is taken as the atom with the smaller index: w/o cutoffs atom1 < atom2 because y<x
        int parent_slot = tree_pointer1;
#endif
        real4 posq1 = posq[atom1];
        real a1 = global_gaussian_exponent[atom1];
        real v1 = global_gaussian_volume[atom1];


#ifdef USE_CUTOFF
        unsigned int j = interactingAtoms[pos*TILE_SIZE + tgx];
        atomIndices[threadIdx.x] = j;
        if(j<PADDED_NUM_ATOMS){
          localData[localAtomIndex].posq = posq[j];
          localData[localAtomIndex].g.w = global_gaussian_exponent[j];
          localData[localAtomIndex].v = global_gaussian_volume[j];
          localData[localAtomIndex].tree_pointer = ovAtomTreePointer[j];
        }
#else
        unsigned int j = x * TILE_SIZE + tgx;
        localData[localAtomIndex].posq = posq[j];
        localData[localAtomIndex].g.w = global_gaussian_exponent[j];
        localData[localAtomIndex].v = global_gaussian_volume[j];
#endif

        SYNC_WARPS;

        unsigned int tj = tgx;
        for (j = 0; j < TILE_SIZE; j++) {

            int localAtom2Index = tbx + tj;
            real4 posq2 = localData[localAtom2Index].posq;
            //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
            real4 delta = make_real4(posq2.x - posq1.x, posq2.y - posq1.y, posq2.z - posq1.z, 0);

            real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            real a2 = localData[localAtom2Index].g.w;
            real v2 = localData[localAtom2Index].v;
#ifdef USE_CUTOFF
            int atom2 = atomIndices[localAtom2Index];
            int tree_pointer2 =  localData[localAtom2Index].tree_pointer;
#else
            int atom2 = x * TILE_SIZE + tj;
#endif
            bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE;
#ifdef USE_CUTOFF
            compute = compute && r2 < CUTOFF_SQUARED;
#else
            //when not using a neighbor list we are getting diagonal tiles here
            if (x == y) compute = compute && atom1 < atom2;
#endif
            if (compute) {
#ifdef USE_CUTOFF
                //the parent is taken as the atom with the smaller index
                int parent_slot = (atom1 < atom2) ? tree_pointer1 : tree_pointer2;
#endif
                COMPUTE_INTERACTION_COUNT
            }
            tj = (tj + 1) & (TILE_SIZE - 1);
            SYNC_WARPS;
        }

        SYNC_WARPS;
        pos++;
    }

}

// version of InitOverlapTreeCount optimized for CPU devices
//  1 CPU core, instead of a minimum of 32 as in the GPU-optimized version, loads a TILE_SIZE of interactions
//  and processes them

//__global__ __attribute__((reqd_work_group_size(1,1,1)))
__device__ void InitOverlapTreeCount_cpu(
        const int *__restrict__ ovAtomTreePointer,    //pointers to atom trees
        const real4 *__restrict__ posq, //atomic positions
        const real *__restrict__ global_gaussian_exponent, //atomic Gaussian exponent
        const real *__restrict__ global_gaussian_volume, //atomic Gaussian volume
#ifdef USE_CUTOFF
const int* __restrict__ tiles,
const unsigned int* __restrict__ interactionCount,
const int* __restrict__ interactingAtoms,
unsigned int maxTiles,
const ushort2* exclusionTiles,
#else
        unsigned int numTiles,
#endif
        int *__restrict__ ovChildrenCount) {

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ncores = blockDim.x * gridDim.x;
    __shared__
    AtomData localData[TILE_SIZE];

    INIT_VARS

    unsigned int warp = id;
    unsigned int totalWarps = ncores;

#ifdef USE_CUTOFF
    //OpenMM's neighbor list stores tiles with exclusions separately from other tiles

    // First loop: process tiles that contain exclusions
    // (this is imposed by OpenMM's neighbor list format, AGBNP does not actually have exclusions)
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
      const ushort2 tileIndices = exclusionTiles[pos];
      unsigned int x = tileIndices.x;
      unsigned int y = tileIndices.y;
      if(y>x) {unsigned int t = y; y = x; x = t;};//swap so that y<x

      // Load the data for this tile in local memory
      for (int j = 0; j < TILE_SIZE; j++) {
        unsigned int atom2 = x*TILE_SIZE + j;
        localData[j].posq = posq[atom2];
        localData[j].g.w = global_gaussian_exponent[atom2];
        localData[j].v = global_gaussian_volume[atom2];
      }

      for (unsigned int tgx = 0; tgx < TILE_SIZE; tgx++) {
        unsigned int atom1 = y*TILE_SIZE+tgx;

        // load atom1 parameters from global arrays
        real4 posq1 = posq[atom1];
        real a1 = global_gaussian_exponent[atom1];
        real v1 = global_gaussian_volume[atom1];
        int parent_slot = ovAtomTreePointer[atom1];

        for (unsigned int j = 0; j < TILE_SIZE; j++) {
      unsigned int atom2 = x*TILE_SIZE+j;

      // load atom2 parameters from local arrays
      real4 posq2 = localData[j].posq;
      real a2 = localData[j].g.w;
      real v2 = localData[j].v;

      //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real4 delta = make_real4(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z, 0);
      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

      bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && atom1 < atom2 && r2 < CUTOFF_SQUARED;
      if (compute) {
        COMPUTE_INTERACTION_COUNT
          }
        }
      }
    }
#endif //USE_CUTOFF


    //second loop, tiles without exclusions or all interactions if not using cutoffs
#ifdef USE_CUTOFF
    __shared__ int atomIndices[TILE_SIZE];
    unsigned int numTiles = interactionCount[0];
    if(numTiles > maxTiles)
      return; // There wasn't enough memory for the neighbor list.
#endif
    int pos = (int) (warp * (long) numTiles / totalWarps);
    int end = (int) ((warp + 1) * (long) numTiles / totalWarps);
    while (pos < end) {
#ifdef USE_CUTOFF
        // y-atom block of the tile
        // atoms in x-atom block (y <= x) are retrieved from interactingAtoms[] below
        unsigned int y = tiles[pos];
#else
        // find x and y coordinates of the tile such that y <= x
        int y = (int) floor(NUM_BLOCKS + 0.5f - SQRT((NUM_BLOCKS + 0.5f) * (NUM_BLOCKS + 0.5f) - 2 * pos));
        int x = (pos - y * NUM_BLOCKS + y * (y + 1) / 2);
        if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
            y += (x < y ? -1 : 1);
            x = (pos - y * NUM_BLOCKS + y * (y + 1) / 2);
        }
#endif

        // Load the data for this tile in local memory
        for (int localAtomIndex = 0; localAtomIndex < TILE_SIZE; localAtomIndex++) {
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+localAtomIndex];
            atomIndices[localAtomIndex] = j;
            if (j < PADDED_NUM_ATOMS) {
          localData[localAtomIndex].posq = posq[j];
          localData[localAtomIndex].g.w = global_gaussian_exponent[j];
          localData[localAtomIndex].v = global_gaussian_volume[j];
          localData[localAtomIndex].tree_pointer = ovAtomTreePointer[j];
            }
#else
            unsigned int j = x * TILE_SIZE + localAtomIndex;
            localData[localAtomIndex].posq = posq[j];
            localData[localAtomIndex].g.w = global_gaussian_exponent[j];
            localData[localAtomIndex].v = global_gaussian_volume[j];
#endif
        }

        for (unsigned int tgx = 0; tgx < TILE_SIZE; tgx++) {
            unsigned int atom1 = y * TILE_SIZE + tgx;

            // load atom1 parameters from global arrays
            real4 posq1 = posq[atom1];
            real a1 = global_gaussian_exponent[atom1];
            real v1 = global_gaussian_volume[atom1];
            int tree_pointer1 = ovAtomTreePointer[atom1];
#ifndef USE_CUTOFF
            //the parent is taken as the atom with the smaller index: w/o cutoffs atom1 < atom2 because y<x
            int parent_slot = tree_pointer1;
#endif

            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                // load atom2 parameters from local arrays
                real4 posq2 = localData[j].posq;
                real a2 = localData[j].g.w;
                real v2 = localData[j].v;

                //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
                real4 delta = make_real4(posq2.x - posq1.x, posq2.y - posq1.y, posq2.z - posq1.z, 0);
                real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
#ifdef USE_CUTOFF
                unsigned int atom2 = atomIndices[j];
                int tree_pointer2 =  localData[j].tree_pointer;
#else
                unsigned int atom2 = x * TILE_SIZE + j;
#endif
                bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE;
#ifdef USE_CUTOFF
                compute = compute && r2 < CUTOFF_SQUARED;
#else
                //when not using a neighbor list we are getting diagonal tiles here
                if (x == y) compute = compute && atom1 < atom2;
#endif
                if (compute) {
#ifdef USE_CUTOFF
                    //the parent is taken as the atom with the smaller index
                    int parent_slot = (atom1 < atom2) ? tree_pointer1 : tree_pointer2;
#endif
                    COMPUTE_INTERACTION_COUNT
                }
            }
        }
        pos++;
    }

}


//this kernel counts the no. of 2-body overlaps for each atom, stores in ovChildrenCount
extern "C" __global__ void InitOverlapTree(
        const int *__restrict__ ovAtomTreePointer,    //pointers to atom trees
        int *__restrict__ ovAtomTreeSize,       //actual sizes
        const int *__restrict__ ovAtomTreePaddedSize, //padded allocated sizes
        const real4 *__restrict__ posq, //atomic positions
        const real *__restrict__ global_gaussian_exponent, //atomic Gaussian exponent
        const real *__restrict__ global_gaussian_volume, //atomic Gaussian volume
        const real *__restrict__ global_atomic_gamma, //atomic gammas
#ifdef USE_CUTOFF
const int* __restrict__ tiles,
const unsigned int* __restrict__ interactionCount,
const int* __restrict__ interactingAtoms,
unsigned int maxTiles,
const ushort2* exclusionTiles,
#else
        unsigned int numTiles,
#endif
        int *__restrict__ ovLevel, //this and below define tree
        real *__restrict__ ovVolume,
        real *__restrict__ ovVsp,
        real *__restrict__ ovVSfp,
        real *__restrict__ ovGamma1i,
        real4 *__restrict__ ovG,
        real4 *__restrict__ ovDV1,

        int *__restrict__ ovLastAtom,
        int *__restrict__ ovRootIndex,
        int *__restrict__ ovChildrenStartIndex,
        int *__restrict__ ovChildrenCount,
        int *__restrict__ ovChildrenCountTop,
        int *__restrict__ ovChildrenCountBottom,
        int *__restrict__ PanicButton) {
    const unsigned int totalWarps = blockDim.x * gridDim.x / TILE_SIZE;
    const unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const unsigned int tgx = threadIdx.x & (TILE_SIZE - 1); //warp id in group
    const unsigned int tbx = threadIdx.x - tgx;           //id in warp
    __shared__
    AtomData localData[FORCE_WORK_GROUP_SIZE];
    const unsigned int localAtomIndex = threadIdx.x;

    INIT_VARS

    if (PanicButton[0] > 0) return;

#ifdef USE_CUTOFF
    //OpenMM's neighbor list stores tiles with exclusions separately from other tiles

    // First loop: process tiles that contain exclusions
    // (this is imposed by OpenMM's neighbor list format, GKCavitation does not actually have exclusions)
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
      const ushort2 tileIndices = exclusionTiles[pos];
      unsigned int x = tileIndices.x;
      unsigned int y = tileIndices.y;
      if(y>x) {unsigned int t = y; y = x; x = t;};//swap so that y<x

      unsigned int atom1 = y*TILE_SIZE + tgx;
      int parent_slot = ovAtomTreePointer[atom1];
      int parent_children_start = ovChildrenStartIndex[parent_slot];

      // Load atom data for this tile.
      real4 posq1 = posq[atom1];
      LOAD_ATOM1_PARAMETERS

      unsigned int j = x*TILE_SIZE + tgx;
      localData[localAtomIndex].posq = posq[j];
      LOAD_LOCAL_PARAMETERS_FROM_GLOBAL

      SYNC_WARPS;

      if(y==x){//diagonal tile

        unsigned int tj = tgx;
        for (j = 0; j < TILE_SIZE; j++) {

      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real4 delta = make_real4(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z, 0);

      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      LOAD_ATOM2_PARAMETERS
        int atom2 = x*TILE_SIZE+tj;

      if (atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && atom1 < atom2 && r2 < CUTOFF_SQUARED) {
        int child_atom = atom2;
        COMPUTE_INTERACTION_STORE1
          }
      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
        }

     }else{

        unsigned int tj = tgx;
        for (j = 0; j < TILE_SIZE; j++) {

      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real4 delta = make_real4(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z, 0);

      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      LOAD_ATOM2_PARAMETERS
        int atom2 = x*TILE_SIZE+tj;

      if (atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && r2 < CUTOFF_SQUARED) {
        int child_atom = atom2;
        COMPUTE_INTERACTION_STORE1
          }
      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
        }

      }

      SYNC_WARPS;
    }
#endif //USE_CUTOFF


    //second loop, tiles without exclusions or all interactions if not using cutoffs
#ifdef USE_CUTOFF
    __shared__ int atomIndices[FORCE_WORK_GROUP_SIZE];
    unsigned int numTiles = interactionCount[0];
    if(numTiles > maxTiles)
      return; // There wasn't enough memory for the neighbor list.
#endif
    int pos = (int) (warp * (long) numTiles / totalWarps);
    int end = (int) ((warp + 1) * (long) numTiles / totalWarps);
    while (pos < end) {
#ifdef USE_CUTOFF
        // y-atom block of the tile
        // atoms in x-atom block (y <= x?) are retrieved from interactingAtoms[] below
        unsigned int y = tiles[pos];
        //unsigned int iat = y*TILE_SIZE + tgx;
        //unsigned int jat = interactingAtoms[pos*TILE_SIZE + tgx];
#else
        // find x and y coordinates of the tile such that y <= x
        int y = (int) floor(NUM_BLOCKS + 0.5f - SQRT((NUM_BLOCKS + 0.5f) * (NUM_BLOCKS + 0.5f) - 2 * pos));
        int x = (pos - y * NUM_BLOCKS + y * (y + 1) / 2);
        if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
            y += (x < y ? -1 : 1);
            x = (pos - y * NUM_BLOCKS + y * (y + 1) / 2);
        }
#endif

        unsigned int atom1 = y * TILE_SIZE + tgx;

        // Load atom data for this tile.
        int tree_pointer1 = ovAtomTreePointer[atom1];
#ifndef USE_CUTOFF
        //the parent is taken as the atom with the smaller index: w/o cutoffs atom1 < atom2 because y<x
        int parent_slot = tree_pointer1;
        int parent_children_start = ovChildrenStartIndex[parent_slot];
#endif
        real4 posq1 = posq[atom1];
        LOAD_ATOM1_PARAMETERS

#ifdef USE_CUTOFF
        unsigned int j = interactingAtoms[pos*TILE_SIZE + tgx];
        atomIndices[threadIdx.x] = j;
        if(j<PADDED_NUM_ATOMS){
        localData[localAtomIndex].posq = posq[j];
        localData[localAtomIndex].tree_pointer = ovAtomTreePointer[j];
        LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
        }
#else
        unsigned int j = x * TILE_SIZE + tgx;
        localData[localAtomIndex].posq = posq[j];
        LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
#endif
        localData[localAtomIndex].ov_count = 0;

        SYNC_WARPS;

        unsigned int tj = tgx;
        for (j = 0; j < TILE_SIZE; j++) {

            int localAtom2Index = tbx + tj;
            real4 posq2 = localData[localAtom2Index].posq;
            //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
            real4 delta = make_real4(posq2.x - posq1.x, posq2.y - posq1.y, posq2.z - posq1.z, 0);
            real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            LOAD_ATOM2_PARAMETERS
#ifdef USE_CUTOFF
            int atom2 = atomIndices[localAtom2Index];
            int tree_pointer2 = localData[localAtom2Index].tree_pointer;
#else
            int atom2 = x * TILE_SIZE + tj;
#endif
            bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE;
#ifdef USE_CUTOFF
            compute = compute && r2 < CUTOFF_SQUARED;
#else
            //when not using a neighbor list we are getting diagonal tiles here
            if (x == y) compute = compute && atom1 < atom2;
#endif
            if (compute) {
#ifdef USE_CUTOFF
                //the parent is taken as the atom with the smaller index
                bool ordered = atom1 < atom2;
                int parent_slot = (ordered) ? tree_pointer1 : tree_pointer2;
                int parent_children_start = ovChildrenStartIndex[parent_slot];
                int child_atom = (ordered) ? atom2 : atom1 ;
                if(!ordered) delta = -delta; //parent and child are reversed (atom2>atom1)
#else
                int child_atom = atom2;
#endif
                COMPUTE_INTERACTION_STORE1
            }
            tj = (tj + 1) & (TILE_SIZE - 1);
            SYNC_WARPS;
        }

        SYNC_WARPS;
        pos++;
    }

}


// version of InitOverlapTreeCount optimized for CPU devices
//  1 CPU core, instead of 32 as in the GPU-optimized version, loads a TILE_SIZE of interactions
//  and process them

//__global__ __attribute__((reqd_work_group_size(1,1,1)))
__device__ void InitOverlapTree_cpu(const int *__restrict__ ovAtomTreePointer,    //pointers to atom trees
                                    int *__restrict__ ovAtomTreeSize,       //actual sizes
                                    const int *__restrict__ ovAtomTreePaddedSize, //padded allocated sizes
                                    const real4 *__restrict__ posq, //atomic positions
                                    const real *__restrict__ global_gaussian_exponent, //atomic Gaussian exponent
                                    const real *__restrict__ global_gaussian_volume, //atomic Gaussian volume
                                    const real *__restrict__ global_atomic_gamma, //atomic gammas
#ifdef USE_CUTOFF
const int* __restrict__ tiles,
const unsigned int* __restrict__ interactionCount,
const int* __restrict__ interactingAtoms,
unsigned int maxTiles,
const ushort2* exclusionTiles,
#else
                                    unsigned int numTiles,
#endif
                                    int *__restrict__ ovLevel, //this and below define tree
                                    real *__restrict__ ovVolume,
                                    real *__restrict__ ovVsp,
                                    real *__restrict__ ovVSfp,
                                    real *__restrict__ ovGamma1i,
                                    real4 *__restrict__ ovG,
                                    real4 *__restrict__ ovDV1,

                                    int *__restrict__ ovLastAtom,
                                    int *__restrict__ ovRootIndex,
                                    int *__restrict__ ovChildrenStartIndex,
                                    int *__restrict__ ovChildrenCount,
                                    int *__restrict__ ovChildrenCountTop,
                                    int *__restrict__ ovChildrenCountBottom,
                                    int *__restrict__ PanicButton) {

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ncores = blockDim.x * gridDim.x;
    __shared__
    AtomData localData[TILE_SIZE];

    INIT_VARS

    unsigned int warp = id;
    unsigned int totalWarps = ncores;

    if (PanicButton[0] > 0) return;

#ifdef USE_CUTOFF
    //OpenMM's neighbor list stores tiles with exclusions separately from other tiles

    // First loop: process tiles that contain exclusions
    // (this is imposed by OpenMM's neighbor list format, AGBNP does not actually have exclusions)
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
      const ushort2 tileIndices = exclusionTiles[pos];
      unsigned int x = tileIndices.x;
      unsigned int y = tileIndices.y;
      if(y>x) {unsigned int t = y; y = x; x = t;};//swap so that y<x

      // Load the data for this tile in local memory
      for (int j = 0; j < TILE_SIZE; j++) {
        unsigned int atom2 = x*TILE_SIZE + j;
        localData[j].posq = posq[atom2];
        localData[j].g.w = global_gaussian_exponent[atom2];
        localData[j].v = global_gaussian_volume[atom2];
        localData[j].gamma = global_atomic_gamma[atom2];
      }

      for (unsigned int tgx = 0; tgx < TILE_SIZE; tgx++) {
        unsigned int atom1 = y*TILE_SIZE+tgx;

        // load atom1 parameters from global arrays
        real4 posq1 = posq[atom1];
        real a1 = global_gaussian_exponent[atom1];
        real v1 = global_gaussian_volume[atom1];
        real gamma1 = global_atomic_gamma[atom1];

        int parent_slot = ovAtomTreePointer[atom1];
        int parent_children_start = ovChildrenStartIndex[parent_slot];

        for (unsigned int j = 0; j < TILE_SIZE; j++) {
      unsigned int atom2 = x*TILE_SIZE+j;

      // load atom2 parameters from local arrays
      real4 posq2 = localData[j].posq;
      real a2 = localData[j].g.w;
      real v2 = localData[j].v;
      real gamma2 = localData[j].gamma;

      //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real4 delta = make_real4(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z, 0);
      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

      bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && r2 < CUTOFF_SQUARED;
      //for diagonal tile make sure that each pair is processed only once
      if(y==x) compute = compute && atom1 < atom2;
      if (compute) {
        int child_atom = atom2;
        COMPUTE_INTERACTION_STORE1
       }
        }
      }
    }
#endif //USE_CUTOFF

    //second loop, tiles without exclusions or all interactions if not using cutoffs
#ifdef USE_CUTOFF
    __shared__ int atomIndices[TILE_SIZE];
    unsigned int numTiles = interactionCount[0];
    if(numTiles > maxTiles)
      return; // There wasn't enough memory for the neighbor list.
#endif
    int pos = (int) (warp * (long) numTiles / totalWarps);
    int end = (int) ((warp + 1) * (long) numTiles / totalWarps);
    while (pos < end) {
#ifdef USE_CUTOFF
        // y-atom block of the tile
        // atoms in x-atom block (y <= x) are retrieved from interactingAtoms[] below
        unsigned int y = tiles[pos];
#else
        // find x and y coordinates of the tile such that y <= x
        int y = (int) floor(NUM_BLOCKS + 0.5f - SQRT((NUM_BLOCKS + 0.5f) * (NUM_BLOCKS + 0.5f) - 2 * pos));
        int x = (pos - y * NUM_BLOCKS + y * (y + 1) / 2);
        if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
            y += (x < y ? -1 : 1);
            x = (pos - y * NUM_BLOCKS + y * (y + 1) / 2);
        }
#endif

        // Load the data for this tile in local memory
        for (int localAtomIndex = 0; localAtomIndex < TILE_SIZE; localAtomIndex++) {
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+localAtomIndex];
            atomIndices[localAtomIndex] = j;
            if (j < PADDED_NUM_ATOMS) {
          localData[localAtomIndex].posq = posq[j];
          localData[localAtomIndex].g.w = global_gaussian_exponent[j];
          localData[localAtomIndex].v = global_gaussian_volume[j];
          localData[localAtomIndex].gamma = global_atomic_gamma[j];
          localData[localAtomIndex].tree_pointer = ovAtomTreePointer[j];
            }
#else
            unsigned int j = x * TILE_SIZE + localAtomIndex;
            localData[localAtomIndex].posq = posq[j];
            localData[localAtomIndex].g.w = global_gaussian_exponent[j];
            localData[localAtomIndex].v = global_gaussian_volume[j];
            localData[localAtomIndex].gamma = global_atomic_gamma[j];
#endif
            localData[localAtomIndex].ov_count = 0;
        }

        for (unsigned int tgx = 0; tgx < TILE_SIZE; tgx++) {
            unsigned int atom1 = y * TILE_SIZE + tgx;

            // load atom1 parameters from global arrays
            real4 posq1 = posq[atom1];
            real a1 = global_gaussian_exponent[atom1];
            real v1 = global_gaussian_volume[atom1];
            real gamma1 = global_atomic_gamma[atom1];
            int tree_pointer1 = ovAtomTreePointer[atom1];
#ifndef USE_CUTOFF
            //the parent is taken as the atom with the smaller index: w/o cutoffs atom1 < atom2 because y<x
            int parent_slot = tree_pointer1;
            int parent_children_start = ovChildrenStartIndex[parent_slot];
#endif

            for (unsigned int j = 0; j < TILE_SIZE; j++) {

                // load atom2 parameters from local arrays
                real4 posq2 = localData[j].posq;
                real a2 = localData[j].g.w;
                real v2 = localData[j].v;
                real gamma2 = localData[j].gamma;

                //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
                real4 delta = make_real4(posq2.x - posq1.x, posq2.y - posq1.y, posq2.z - posq1.z, 0);
                real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

#ifdef USE_CUTOFF
                unsigned int atom2 = atomIndices[j];
                int tree_pointer2 = localData[j].tree_pointer;
#else
                unsigned int atom2 = x * TILE_SIZE + j;
#endif
                bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE;
#ifdef USE_CUTOFF
                compute = compute && r2 < CUTOFF_SQUARED;
#else
                //when not using a neighbor list we are getting diagonal tiles here
                if (x == y) compute = compute && atom1 < atom2;
#endif
                if (compute) {
#ifdef USE_CUTOFF
                    //the parent is taken as the atom with the smaller index
                    bool ordered = atom1 < atom2;
                    int parent_slot = (ordered) ? tree_pointer1 : tree_pointer2;
                    int parent_children_start = ovChildrenStartIndex[parent_slot];
                    int child_atom = (ordered) ? atom2 : atom1 ;
                    if(!ordered) delta = -delta; //parent and child are reversed (atom2>atom1)
#else
                    int child_atom = atom2;
#endif
                    COMPUTE_INTERACTION_STORE1
                }

            }
        }
        pos++;
    }
}


//this kernel initializes the tree to be processed by ComputeOverlapTree()
//it assumes that 2-body overlaps are in place
extern "C" __global__ void resetComputeOverlapTree(const int ntrees,
                                                   const int *__restrict__ ovTreePointer,
                                                   int *__restrict__ ovProcessedFlag,
                                                   int *__restrict__ ovOKtoProcessFlag,
                                                   const int *__restrict__ ovAtomTreeSize,
                                                   const int *__restrict__ ovLevel
) {
    unsigned int local_id = threadIdx.x;
    int tree = blockIdx.x;
    while (tree < ntrees) {
        unsigned int tree_ptr = ovTreePointer[tree];
        unsigned int tree_size = ovAtomTreeSize[tree];
        unsigned int endslot = tree_ptr + tree_size;
        unsigned int slot = tree_ptr + local_id;
        while (slot < endslot) {
            if (ovLevel[slot] == 1) {
                ovProcessedFlag[slot] = 1;
                ovOKtoProcessFlag[slot] = 0;
            } else if (ovLevel[slot] == 2) {
                ovProcessedFlag[slot] = 0; //flag 2-body overlaps as ready for processing.
                ovOKtoProcessFlag[slot] = 1;
            }
            slot += blockDim.x;
        }
        tree += gridDim.x;
        //TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
    }
}

//===================================================
//Utilities to do parallel prefix sum of an array ("scan").
//The input is an integer array and the output is the sum of
//the elements up to that element:
//a[i] -> Sum(j=0)to(j=i) a[j] ("inclusive")
//a[i] -> Sum(j=0)to(j=i-1) a[j] ("exclusive")
//
//Derived from NVIDIA's GPU Computing SDK
//https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclMarchingCubes/Scan.cl
//
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
//
//Note that the input is the item corresponding to the current thread (not the array)
//the output can be extracted from the second half of l_Data[] (I think)
//or directly from the return value (current thread)
//This version works only for a single work group so it is limited
//to array sizes = to max work group size
__device__ inline unsigned int scan1Inclusive(unsigned int idata,
                                              unsigned int *l_Data,
                                              unsigned int size) {
    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    l_Data[pos] = 0;
    pos += size;
    l_Data[pos] = idata;
    __syncthreads();

    for (unsigned int offset = 1; offset < size; offset <<= 1) {
        __syncthreads();
        int t = l_Data[pos] + l_Data[pos - offset];
        __syncthreads();
        l_Data[pos] = t;
    }

    __syncthreads();
    return l_Data[pos];
}

__device__ inline unsigned int scan1Exclusive(unsigned int idata,
                                              unsigned int *l_Data,
                                              unsigned int size) {
    return scan1Inclusive(idata, l_Data, size) - idata;
}


//scan of general size over global buffers (size must be multiple of work group size)
//repeated application of scan over work-group chunks
__device__ inline void scangExclusive(unsigned int *buffer,
                                      unsigned int *l_Data,
                                      unsigned int size) {
    unsigned int gsize = blockDim.x;
    unsigned int niter = size / gsize;
    unsigned int id = threadIdx.x;
    __shared__ unsigned int psum;

    unsigned int i = id;

    unsigned int sum = scan1Exclusive(buffer[i], l_Data, gsize);
    if (id == gsize - 1) psum = sum + buffer[i];
    buffer[i] = sum;
//TODOLater: Global memory fence needed or syncthreads sufficient?
    __syncthreads();
    i += gsize;

    while (i < size) {
        unsigned int sum = scan1Exclusive(buffer[i], l_Data, gsize) + psum;
        __syncthreads();
        if (id == gsize - 1) psum = sum + buffer[i];
        buffer[i] = sum;
//TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
        i += gsize;
    }
    __syncthreads();
}

//=====================================================================


//used for the "scan" of children counts to get children start indexes
//assumes that work group size = OV_WORK_GROUP_SIZE

//__global__ __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
extern "C" __global__ void reduceovCountBuffer(const int ntrees,
                                               const int* __restrict__ ovTreePointer,
                                               const int* __restrict__ ovAtomTreePointer,    //pointers to atom trees
                                               int* __restrict__ ovAtomTreeSize,       //actual sizes
                                               const int* __restrict__ ovAtomTreePaddedSize, //actual sizes
                                               unsigned int* __restrict__ ovChildrenStartIndex,
                                               int* __restrict__ ovChildrenCount,
                                               int* __restrict__ ovChildrenCountTop,
                                               int* __restrict__ ovChildrenCountBottom,
                                               int* __restrict__ PanicButton) {
    unsigned int local_id = threadIdx.x;
    unsigned int gsize = blockDim.x;
    __shared__ unsigned int temp[2 * OV_WORK_GROUP_SIZE];

    int tree = blockIdx.x;
    while (tree < ntrees && PanicButton[0] == 0) {

        unsigned int tree_size = ovAtomTreeSize[tree];
        unsigned int tree_ptr = ovTreePointer[tree];

        if (tree_size <= gsize) {

            // number of 1-body overlaps is less than
            // group size, can use faster scan routine

            unsigned int atom_ptr = tree_ptr + local_id;
            int children_count = 0;
            if (local_id < tree_size) {
                children_count = ovChildrenCount[atom_ptr];
            }
            unsigned int sum = scan1Exclusive(children_count, temp, gsize);
            __syncthreads();
            if (local_id < tree_size) {
                ovChildrenStartIndex[atom_ptr] = tree_ptr + tree_size + sum;
                // resets to top and bottom counters
                ovChildrenCountTop[atom_ptr] = 0;
                ovChildrenCountBottom[atom_ptr] = 0;
            }
            if (local_id == tree_size - 1) {
                //update tree size to include 2-body
                ovAtomTreeSize[tree] += sum + children_count;
            }

        } else {

            // do scan of an array of arbitrary size

            unsigned int padded_tree_size = gsize * ((tree_size + gsize - 1) / gsize);
            for (unsigned int i = local_id; i < padded_tree_size; i += gsize) {
                unsigned int atom_ptr = tree_ptr + i;
                ovChildrenStartIndex[atom_ptr] = (i < tree_size) ? ovChildrenCount[atom_ptr] : 0;
            }
            //TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads();

            scangExclusive(&(ovChildrenStartIndex[tree_ptr]), temp, padded_tree_size);
            //TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads();
            if (local_id == 0) {
                ovAtomTreeSize[tree] +=
                        ovChildrenStartIndex[tree_ptr + tree_size - 1] + ovChildrenCount[tree_ptr + tree_size - 1];
            }
            //TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads();
            for (unsigned int i = local_id; i < padded_tree_size; i += gsize) {
                if (i < tree_size) {
                    ovChildrenStartIndex[tree_ptr + i] += tree_ptr + tree_size;
                } else {
                    ovChildrenStartIndex[tree_ptr + i] = 0;
                }
            }
            for (unsigned int atom_ptr = tree_ptr + local_id;
                 atom_ptr < tree_ptr + padded_tree_size; atom_ptr += gsize) {
                ovChildrenCountTop[atom_ptr] = 0;
            }
            for (unsigned int atom_ptr = tree_ptr + local_id;
                 atom_ptr < tree_ptr + padded_tree_size; atom_ptr += gsize) {
                ovChildrenCountBottom[atom_ptr] = 0;
            }
        }
        if (local_id == 0 && ovAtomTreeSize[tree] >= ovAtomTreePaddedSize[tree]) {
            //atomic_inc(&PanicButton[0]);
            atomicAdd(&PanicButton[0], 1);
        }
        //next tree
        tree += gridDim.x;
        //TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
    }

}

// insertion sort
__device__ inline void sortVolumes2body(unsigned const int idx,
                                        unsigned const int nx,
                                        real *__restrict__ ovVolume,
                                        real *__restrict__ ovVSfp,
                                        real *__restrict__ ovGamma1i,
                                        real4 *__restrict__ ovG,
                                        real4 *__restrict__ ovDV1,
                                        int *__restrict__ ovLastAtom) {

    if (nx > 0) {

        for (unsigned int k = idx + 1; k < idx + nx; k++) {

            real v = ovVolume[k];
            real sfp = ovVSfp[k];
            real gamma = ovGamma1i[k];
            real4 g4 = ovG[k];
            real4 dv1 = ovDV1[k];
            int atom = ovLastAtom[k];

            unsigned int j = k - 1;
            while (j >= idx && ovVolume[j] < v) {
                ovVolume[j + 1] = ovVolume[j];
                ovVSfp[j + 1] = ovVSfp[j];
                ovGamma1i[j + 1] = ovGamma1i[j];
                ovG[j + 1] = ovG[j];
                ovDV1[j + 1] = ovDV1[j];
                ovLastAtom[j + 1] = ovLastAtom[j];
                j -= 1;
            }
            ovVolume[j + 1] = v;
            ovVSfp[j + 1] = sfp;
            ovGamma1i[j + 1] = gamma;
            ovG[j + 1] = g4;
            ovDV1[j + 1] = dv1;
            ovLastAtom[j + 1] = atom;
        }
    }
}

/* this kernel sorts the 2-body portions of the tree according to volume. It is structured so that each thread gets one atom */
extern "C" __global__ void SortOverlapTree2body(const int *__restrict__ ovAtomTreePointer,    //pointers to atom trees
                                                const int *__restrict__ ovAtomTreeSize,       //actual sizes
                                                const int *__restrict__ ovAtomTreePaddedSize, //padded allocated sizes
                                                int *__restrict__ ovLevel, //this and below define tree
                                                real *__restrict__ ovVolume,
                                                real *__restrict__ ovVSfp,
                                                real *__restrict__ ovGamma1i,
                                                real4 *__restrict__ ovG,
                                                real4 *__restrict__ ovDV1,
                                                int *__restrict__ ovLastAtom,
                                                int *__restrict__ ovRootIndex,
                                                const int *__restrict__ ovChildrenStartIndex,
                                                const int *__restrict__ ovChildrenCount
) {
    unsigned int atom = blockIdx.x * blockDim.x + threadIdx.x; // to start

    while (atom < NUM_ATOMS_TREE) {

        unsigned int atom_ptr = ovAtomTreePointer[atom];
        int size = ovChildrenCount[atom_ptr];
        int offset = ovChildrenStartIndex[atom_ptr];

        if (size > 0 && offset >= 0 && ovLastAtom[atom_ptr] >= 0) {
            // sort 2-body volumes of atom
            sortVolumes2body(offset, size,
                             ovVolume,
                             ovVSfp,
                             ovGamma1i,
                             ovG,
                             ovDV1,
                             ovLastAtom);
        }
        atom += blockDim.x * gridDim.x;
    }
//TODOLater: Global memory fence needed or syncthreads sufficient?
    __syncthreads();
}

//this kernel completes the tree with 3-body and higher overlaps avoiding 2 passes over overlap volumes.
//Each workgroup of size OV_WORK_GROUP_SIZE is assigned to a tree section
//Then starting from the top of the tree section, for each slot i to process do:
//1. Retrieve siblings j and process overlaps i<j, count non-zero overlaps
//2. Do a parallel prefix sum (scan) of the array of counts to fill ovChildrenStartIndex[]
//3. Re-process the i<j overlaps and saves them starting at ovChildrenStartIndex[]

//__global__ __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
extern "C" __global__ void ComputeOverlapTree_1pass(const int ntrees,
                                                    const int *__restrict__ ovTreePointer,
                                                    const int *__restrict__ ovAtomTreePointer,    //pointers to atom trees
                                                    int *__restrict__ ovAtomTreeSize,       //actual sizes
                                                    int *__restrict__ NIterations,
                                                    const int *__restrict__ ovAtomTreePaddedSize, //padded allocated sizes
                                                    int *__restrict__ ovAtomTreeLock,       //tree locks
                                                    const real4 *__restrict__ posq, //atomic positions
                                                    const real *__restrict__ global_gaussian_exponent, //atomic Gaussian exponent
                                                    const real *__restrict__ global_gaussian_volume, //atomic Gaussian prefactor
                                                    const real *__restrict__ global_atomic_gamma, //atomic Gaussian prefactor
                                                    int *__restrict__ ovLevel, //this and below define tree
                                                    real *__restrict__ ovVolume,
                                                    real *__restrict__ ovVsp,
                                                    real *__restrict__ ovVSfp,
                                                    real *__restrict__ ovGamma1i,
                                                    real4 *__restrict__ ovG,
                                                    real4 *__restrict__ ovDV1,
                                                    int *__restrict__ ovLastAtom,
                                                    int *__restrict__ ovRootIndex,
                                                    int *__restrict__ ovChildrenStartIndex,
                                                    int *__restrict__ ovChildrenCount,
                                                    volatile int *__restrict__ ovProcessedFlag,
                                                    volatile int *__restrict__ ovOKtoProcessFlag,
                                                    volatile int *__restrict__ ovChildrenReported,
                                                    int *__restrict__ ovChildrenCountTop,
                                                    int *__restrict__ ovChildrenCountBottom,
        // temporary buffers
                                                    unsigned const int buffer_size,//assume buffer_size/ntrees = multiple of group size
                                                    real *__restrict__ gvol_buffer,
                                                    unsigned int *__restrict__ tree_pos_buffer, // where to store in tree
                                                    int *__restrict__ i_buffer,
                                                    int *__restrict__ atomj_buffer,
                                                    int *__restrict__ PanicButton) {

    const unsigned int local_id = threadIdx.x;
    __shared__ unsigned int temp[2 * OV_WORK_GROUP_SIZE];
    __shared__ volatile unsigned int nprocessed;
    __shared__ volatile unsigned int tree_size;
    __shared__ volatile unsigned int niterations;

    const unsigned int gsize = OV_WORK_GROUP_SIZE;

    const unsigned int max_level = MAX_ORDER;

    __shared__ volatile unsigned int panic;//flag to interrupt calculation
    __shared__ volatile unsigned int n_buffer; //how many overlaps in buffer to process
    __shared__ volatile unsigned int buffer_pos[OV_WORK_GROUP_SIZE]; //where to store in temp. buffers
    __shared__ volatile int parent1_buffer[OV_WORK_GROUP_SIZE]; //tree slot of "i" overlap
    __shared__ volatile int level1_buffer[OV_WORK_GROUP_SIZE]; //overlap level of of "i" overlap
    //TODO: May need to declare posq1_buffer as volatile
    __shared__ real4 posq1_buffer[OV_WORK_GROUP_SIZE]; //position of "i" overlap
    __shared__ volatile real a1_buffer[OV_WORK_GROUP_SIZE]; //a parameter of "i" overlap
    __shared__ volatile real v1_buffer[OV_WORK_GROUP_SIZE]; //volume of "i" overlap
    __shared__ volatile real gamma1_buffer[OV_WORK_GROUP_SIZE]; //gamma parameter of "i" overlap
    __shared__ volatile int children_count[OV_WORK_GROUP_SIZE]; //number of children

    if (local_id == 0) panic = PanicButton[0];
    __syncthreads();

    unsigned int tree = blockIdx.x;
    while (tree < ntrees && panic == 0) {
        unsigned int tree_ptr = ovTreePointer[tree];
        unsigned int buffer_offset = tree * (buffer_size / ntrees);

        //initializes local working copy of tree size;
        if (local_id == OV_WORK_GROUP_SIZE - 1) tree_size = ovAtomTreeSize[tree];
        __syncthreads();

        //this is the number of translations of the calculations window
        //to cover the tree section
        //make sure padded tree size is multiple of OV_WORK_GROUP_SIZE
        const unsigned int nsections = ovAtomTreePaddedSize[tree] / gsize;
        //start at the top of the tree and iterate until the end of the tree is reached
        for (unsigned int isection = 0; isection < nsections; isection++) {
            unsigned int slot = tree_ptr + isection * OV_WORK_GROUP_SIZE + local_id; //the slot to work on

            if (local_id == OV_WORK_GROUP_SIZE - 1) niterations = 0;

            do {
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
                if (local_id == OV_WORK_GROUP_SIZE - 1) nprocessed = 0;

                int parent = ovRootIndex[slot];
                int atom1 = ovLastAtom[slot];
                int processed = ovProcessedFlag[slot];
                int ok2process = ovOKtoProcessFlag[slot];
                int level = ovLevel[slot];
                bool letsgo = (parent >= 0 && processed == 0 && ok2process > 0 && atom1 >= 0 && level < max_level &&
                               panic == 0 && PanicButton[0] == 0);


                //
                // -- phase I: fill-up buffers --
                //

                // step 1: load overlap "i" parameters in local buffers
                level1_buffer[local_id] = level;
                //posq1_buffer[local_id] = (real4)(ovG[slot].xyz,0);
                posq1_buffer[local_id] = make_real4(ovG[slot].x, ovG[slot].y, ovG[slot].z, 0);
                a1_buffer[local_id] = ovG[slot].w;
                v1_buffer[local_id] = ovVolume[slot];
                gamma1_buffer[local_id] = ovGamma1i[slot];
                parent1_buffer[local_id] = slot;
                children_count[local_id] = 0;

                //  step 2: compute buffer pointers and number of overlaps
                unsigned int ov_count = 0;
                int sibling_start = 0;
                int sibling_count = 0;
                if (letsgo) {
                    sibling_start = ovChildrenStartIndex[parent];
                    sibling_count = ovChildrenCount[parent];
                }
                int my_sibling_idx = slot - sibling_start;
                if (letsgo) {
                    // store number of interactions, that is the number" of younger" siblings,
                    // this will undergo a prefix sum below
                    ov_count = sibling_count - (my_sibling_idx + 1);
                }
                unsigned int sum = scan1Exclusive(ov_count, temp, gsize);
                buffer_pos[local_id] = buffer_offset + sum;
                if (local_id == OV_WORK_GROUP_SIZE - 1) {
                    n_buffer = sum + ov_count;
                    if (n_buffer >= buffer_size / ntrees) {
                        //atomic_inc(&panic);
                        //atomic_inc(&PanicButton[0]);
                        //atomic_inc(&PanicButton[1]);
                        atomicAdd((unsigned int *) &panic, 1);
                        atomicAdd(&PanicButton[0], 1);
                        atomicAdd(&PanicButton[1], 1);
                    }
                }
                __syncthreads();

                if (n_buffer > 0 && panic == 0) { //something to process

                    //  step 3: insert data for overlaps "j" into global buffers
                    if (letsgo && sibling_start >= 0 && sibling_count > 0) {
                        int end = sibling_start + sibling_count;
                        // store last atom of j overlap in global buffer
                        unsigned int pos = buffer_pos[local_id];
                        for (int i = sibling_start + my_sibling_idx + 1; i < end; i++, pos++) {
                            i_buffer[pos] = local_id;
                            atomj_buffer[pos] = ovLastAtom[i];
                        }
                    }
//TODOLater: Global memory fence needed or syncthreads sufficient?
                    __syncthreads();

                    //
                    // phase II: compute overlap volumes, compute number of non-zero volumes
                    //           (here threads work in tandem on all overlaps in global buffers)

                    // step 1: compute overlap volumes
                    unsigned int pos = local_id + buffer_offset;
                    while (pos < n_buffer + buffer_offset) {
                        int overlap1 = i_buffer[pos];
                        int atom2 = atomj_buffer[pos];
                        unsigned int fij = 0;
                        real vij = 0;
                        if (atom2 >= 0) {
                            real4 posq1 = posq1_buffer[overlap1];
                            real a1 = a1_buffer[overlap1];
                            real v1 = v1_buffer[overlap1];
                            real4 posq2 = posq[atom2];
                            real a2 = global_gaussian_exponent[atom2];
                            real v2 = global_gaussian_volume[atom2];
                            //Gaussian overlap
                            //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0.0f);
                            real4 delta = make_real4(posq2.x - posq1.x, posq2.y - posq1.y, posq2.z - posq1.z, 0);
                            real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
                            COMPUTE_INTERACTION_GVOLONLY
                                    fij = gvol > VolMinA ? 1 : 0;
                            vij = gvol;
                        }
                        tree_pos_buffer[pos] = fij; //for prefix sum below
                        gvol_buffer[pos] = vij;
                        pos += gsize;
                    }
//TODOLater: Global memory fence needed or syncthreads sufficient?
                    __syncthreads();

                    //step 2: prefix sum over "fij" flag buffer to compute number of non-zero overlaps and
                    //        their placement in the tree.
                    unsigned int padded_n_buffer = ((n_buffer / OV_WORK_GROUP_SIZE) + 1) * OV_WORK_GROUP_SIZE;
                    int np = 0;
                    if (local_id == OV_WORK_GROUP_SIZE - 1) np = tree_pos_buffer[buffer_offset + n_buffer - 1];
                    scangExclusive(&(tree_pos_buffer[buffer_offset]), temp, padded_n_buffer);
//TODOLater: Global memory fence needed or syncthreads sufficient?
                    __syncthreads();
                    if (local_id == OV_WORK_GROUP_SIZE - 1) { //retrieve total number of non-zero overlaps
                        nprocessed = tree_pos_buffer[buffer_offset + n_buffer - 1] + np;
                    }
                    __syncthreads();


                    //step 3: compute other quantities for non-zero volumes and store in tree
                    pos = local_id + buffer_offset;
                    while (pos < n_buffer + buffer_offset && panic == 0) {
                        int overlap1 = i_buffer[pos];
                        int atom2 = atomj_buffer[pos];
                        real gvol = gvol_buffer[pos];
                        unsigned int endslot = tree_ptr + tree_size + tree_pos_buffer[pos];
                        if (endslot - tree_ptr >= ovAtomTreePaddedSize[tree]) {
                            //atomic_inc(&panic);
                            atomicAdd((unsigned int *) &panic, 1);
                        }
                        if (atom2 >= 0 && gvol > VolMinA && panic == 0) {
                            int level = level1_buffer[overlap1] + 1;
                            int parent_slot = parent1_buffer[overlap1];
                            real4 posq1 = posq1_buffer[overlap1];
                            real a1 = a1_buffer[overlap1];
                            real v1 = v1_buffer[overlap1];
                            real gamma1 = gamma1_buffer[overlap1];
                            real4 posq2 = posq[atom2];
                            real a2 = global_gaussian_exponent[atom2];
                            real gamma2 = global_atomic_gamma[atom2];

                            //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0.0f);
                            real4 delta = make_real4(posq2.x - posq1.x, posq2.y - posq1.y, posq2.z - posq1.z, 0);
                            COMPUTE_INTERACTION_OTHER
                                    ovLevel[endslot] = level;
                            ovVolume[endslot] = gvol;
                            ovVsp[endslot] = s;
                            ovVSfp[endslot] = sfp;
                            ovGamma1i[endslot] = gamma1 + gamma2;
                            ovLastAtom[endslot] = atom2;
                            ovRootIndex[endslot] = parent_slot;
                            ovChildrenStartIndex[endslot] = -1;
                            ovChildrenCount[endslot] = 0;
                            //ovG[endslot] = (real4)(c12.xyz, a12);
                            //ovDV1[endslot] = (real4)(-delta.xyz*dgvol,dgvolv);
                            ovG[endslot] = make_real4(c12.x, c12.y, c12.z, a12);
                            ovDV1[endslot] = make_real4(-delta.x * dgvol, -delta.y * dgvol, -delta.z * dgvol, dgvolv);
                            ovProcessedFlag[endslot] = 0;
                            ovOKtoProcessFlag[endslot] = 1;
                            //update parent children counter
                            //atomic_inc(&children_count[overlap1]);
                            atomicAdd((int *) &children_count[overlap1], 1);
                        }
                        pos += gsize;
                    }
                    __syncthreads();

                    //scan of children counts to figure out children start indexes
                    sum = scan1Exclusive(children_count[local_id], temp, gsize);
                    if (letsgo) {
                        if (children_count[local_id] > 0) {
                            ovChildrenStartIndex[slot] = tree_ptr + tree_size + sum;
                            ovChildrenCount[slot] = children_count[local_id];
                        }
                        ovProcessedFlag[slot] = 1;
                        ovOKtoProcessFlag[slot] = 0;
                    }
//TODOLater: Global memory fence needed or syncthreads sufficient?
                    __syncthreads(); //global to sync ovChildrenStartIndex etc.

                    //figures out the new tree size
                    if (local_id == OV_WORK_GROUP_SIZE - 1) {
                        tree_size += nprocessed;
                        niterations += 1;
                    }
                    __syncthreads();

                }//n_buffer > 0
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();

            } while (nprocessed > 0 && niterations < gsize && panic == 0); //matches do{}while

            if (local_id == OV_WORK_GROUP_SIZE - 1) {
                if (niterations > NIterations[tree]) NIterations[tree] = niterations;
            }
//TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads(); //to sync ovProcessedFlag etc.
        }
        __syncthreads();
        //stores tree size in global mem
        if (local_id == 0) ovAtomTreeSize[tree] = tree_size;
//TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();

        //next tree
        tree += gridDim.x;
//TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
    }
    if (local_id == 0 && panic > 0) {
        //atomic_inc(&PanicButton[0]);
        atomicAdd(&PanicButton[0], 1);
    }
//TODOLater: Global memory fence needed or syncthreads sufficient?
    __syncthreads();
}


/**
 * Initialize tree for rescanning, set Processed, OKtoProcess=1 for leaves and out-of-bound,
 */
extern "C" __global__ void ResetRescanOverlapTree(const int ntrees,
                                                  const int *__restrict__ ovTreePointer,
                                                  const int *__restrict__ ovAtomTreePointer,    //pointers to atom tree sections
                                                  const int *__restrict__ ovAtomTreeSize,    //pointers to atom tree sections
                                                  const int *__restrict__ ovAtomTreePaddedSize,    //pointers to atom tree sections
                                                  int *__restrict__ ovProcessedFlag,
                                                  int *__restrict__ ovOKtoProcessFlag) {
    const unsigned int local_id = threadIdx.x;
    const unsigned int gsize = OV_WORK_GROUP_SIZE;

    unsigned int tree = blockIdx.x;
    while (tree < ntrees) {
        unsigned int tree_ptr = ovTreePointer[tree];
        unsigned int tree_size = ovAtomTreePaddedSize[tree];
        unsigned int endslot = tree_ptr + tree_size;

        for (int slot = tree_ptr + local_id; slot < endslot; slot += gsize) {
            ovProcessedFlag[slot] = 0;
        }
        for (int slot = tree_ptr + local_id; slot < endslot; slot += gsize) {
            ovOKtoProcessFlag[slot] = 0;
        }

        //next tree
        tree += gridDim.x;
        //TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
    }

}

//this kernel initializes the tree to be processed by RescanOverlapTree()
extern "C" __global__ void InitRescanOverlapTree(const int ntrees,
                                                 const int *__restrict__ ovTreePointer,
                                                 const int *__restrict__ ovAtomTreeSize,
                                                 int *__restrict__ ovProcessedFlag,
                                                 int *__restrict__ ovOKtoProcessFlag,
                                                 const int *__restrict__ ovLevel) {
    unsigned int local_id = threadIdx.x;
    int tree = blockIdx.x;

    while (tree < ntrees) {
        unsigned int tree_ptr = ovTreePointer[tree];
        unsigned int tree_size = ovAtomTreeSize[tree];
        unsigned int endslot = tree_ptr + tree_size;
        unsigned int slot = tree_ptr + local_id;

        while (slot < endslot) {
            if (ovLevel[slot] == 1) {
                ovProcessedFlag[slot] = 1;
                ovOKtoProcessFlag[slot] = 0;
            } else if (ovLevel[slot] == 2) {
                ovProcessedFlag[slot] = 0; //flag 2-body overlaps as ready for processing.
                ovOKtoProcessFlag[slot] = 1;
            }
            slot += blockDim.x;
        }
        tree += gridDim.x;
        //TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
    }

#ifdef NOTNOW

    if(atom < NUM_ATOMS_TREE){
    int pp = ovAtomTreePointer[atom];
    ovProcessedFlag[pp] = 1; //1body is already done
    ovOKtoProcessFlag[pp] = 0;
    int ic = ovChildrenStartIndex[pp];
    int nc = ovChildrenCount[pp];
    if(ic >= 0 && nc > 0){
      for(int slot = ic; slot < ic+nc ; slot++){
        ovProcessedFlag[slot] = 0; //flag 2-body overlaps as ready for processing.
        ovOKtoProcessFlag[slot] = 1;
      }
    }
    }
    atom += blockDim.x*gridDim.x;
#endif
}

//this kernel recomputes the overlap volumes of the current tree
//it does not modify the tree in any other way

//__global__ __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
extern "C" __global__ void RescanOverlapTree(const int ntrees,
                                             const int *__restrict__ ovTreePointer,
                                             const int *__restrict__ ovAtomTreePointer,    //pointers to atom trees
                                             const int *__restrict__ ovAtomTreeSize,       //actual sizes
                                             int *__restrict__ NIterations,
                                             const int *__restrict__ ovAtomTreePaddedSize, //padded allocated sizes
                                             const int *__restrict__ ovAtomTreeLock,       //tree locks
                                             const real4 *__restrict__ posq, //atomic positions
                                             const real *__restrict__ global_gaussian_exponent, //atomic Gaussian exponent
                                             const real *__restrict__ global_gaussian_volume, //atomic Gaussian prefactor
                                             const real *__restrict__ global_atomic_gamma, //atomic gamma
                                             const int *__restrict__ ovLevel, //this and below define tree
                                             real *__restrict__ ovVolume,
                                             real *__restrict__ ovVsp,
                                             real *__restrict__ ovVSfp,
                                             real *__restrict__ ovGamma1i,
                                             real4 *__restrict__ ovG,
                                             real4 *__restrict__ ovDV1,

                                             const int *__restrict__ ovLastAtom,
                                             const int *__restrict__ ovRootIndex,
                                             const int *__restrict__ ovChildrenStartIndex,
                                             const int *__restrict__ ovChildrenCount,
                                             volatile int *__restrict__ ovProcessedFlag,
                                             volatile int *__restrict__ ovOKtoProcessFlag,
                                             volatile int *__restrict__ ovChildrenReported) {

    const unsigned int local_id = threadIdx.x;
    const unsigned int gsize = OV_WORK_GROUP_SIZE;
    __shared__ unsigned int nprocessed;

    unsigned int tree = blockIdx.x;
    while (tree < ntrees) {
        unsigned int tree_ptr = ovTreePointer[tree];
        unsigned int tree_size = ovAtomTreeSize[tree];

        //this is the number of translations of the calculations window
        //to cover the tree section
        //make sure padded tree size is multiple of OV_WORK_GROUP_SIZE
        const unsigned int nsections = ovAtomTreePaddedSize[tree] / gsize;
        //start at the top of the tree and iterate until the end of the tree is reached
        for (unsigned int isection = 0; isection < nsections; isection++) {
            unsigned int slot = tree_ptr + isection * OV_WORK_GROUP_SIZE + local_id; //the slot to work on
//TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads();
            int parent = ovRootIndex[slot];
            int atom = ovLastAtom[slot];

            //for(unsigned int iiter = 0; iiter < 2 ; iiter++){
            do {
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
                if (local_id == 0) nprocessed = 0;
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
                int processed = ovProcessedFlag[slot];
                int ok2process = ovOKtoProcessFlag[slot];
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
                bool letsgo = (parent >= 0 && processed == 0 && ok2process > 0 && atom >= 0);
                if (letsgo) {
                    //atomic_inc(&nprocessed);
                    atomicAdd(&nprocessed, 1);
                    //real4 posq1 = (real4)(ovG[parent].xyz,0);
                    real4 posq1 = make_real4(ovG[parent].x, ovG[parent].y, ovG[parent].z, 0);
                    real a1 = ovG[parent].w;
                    real v1 = ovVolume[parent];
                    real gamma1 = ovGamma1i[parent];
                    real4 posq2 = posq[atom];
                    real a2 = global_gaussian_exponent[atom];
                    real v2 = global_gaussian_volume[atom];
                    real gamma2 = global_atomic_gamma[atom];
                    //ovGamma1i[slot] = ovGamma1i[parent] + global_atomic_gamma[atom];
                    ovGamma1i[slot] = gamma1 + gamma2;
                    //Gaussian overlap
                    //real4 delta = (real4) (posq2.xyz - posq1.xyz, 0.0f);
                    real4 delta = make_real4(posq2.x - posq1.x, posq2.y - posq1.y, posq2.z - posq1.z, 0);
                    real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
                    COMPUTE_INTERACTION_RESCAN
                    //mark itself as processed and children as okay to process
                            ovProcessedFlag[slot] = 1;
                    ovOKtoProcessFlag[slot] = 0;
                    if (ovChildrenStartIndex[slot] >= 0 && ovChildrenCount[slot] > 0) {
                        for (int i = ovChildrenStartIndex[slot];
                             i < ovChildrenStartIndex[slot] + ovChildrenCount[slot]; i++) {
                            ovOKtoProcessFlag[i] = 1;
                        }
                    }
                }
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
            } while (nprocessed > 0);
//TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads();
        }

        // next tree
        tree += gridDim.x;
//TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
    }
}


//this kernel initializes the 1-body nodes with a new set of atomic gamma parameters
extern "C" __global__ void InitOverlapTreeGammas_1body(unsigned const int num_padded_atoms,
                                                       unsigned const int num_sections,
                                                       const int *__restrict__ ovTreePointer,
                                                       const int *__restrict__ ovNumAtomsInTree,
                                                       const int *__restrict__ ovFirstAtom,
                                                       int *__restrict__ ovAtomTreeSize,    //sizes of tree sections
                                                       int *__restrict__ NIterations,
                                                       const int *__restrict__ ovAtomTreePointer,    //pointers to atoms in tree
                                                       const float *__restrict__ gammaParam, //gamma
                                                       real *__restrict__ ovGamma1i) {
    const unsigned int id = threadIdx.x;

    unsigned int section = blockIdx.x;
    while (section < num_sections) {
        int natoms_in_section = ovNumAtomsInTree[section];
        int iat = id;
        while (iat < natoms_in_section) {
            int atom = ovFirstAtom[section] + iat;

            real g = gammaParam[atom];
            int slot = ovAtomTreePointer[atom];
            ovGamma1i[slot] = g;

            iat += blockDim.x;
        }
        if (id == 0) {
            NIterations[section] = 0;
        }

        section += gridDim.x;
//TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
    }
}


//Same as RescanOverlapTree above:
//propagates gamma atomic parameters from the top to the bottom
//of the overlap tree,
//it *does not* recompute overlap volumes
//  used to prep calculations of volume derivatives of GB and van der Waals energies

//__global__ __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
extern "C" __global__ void RescanOverlapTreeGammas(const int ntrees,
                                                   const int *__restrict__ ovTreePointer,
                                                   const int *__restrict__ ovAtomTreePointer,    //pointers to atom trees
                                                   const int *__restrict__ ovAtomTreeSize,       //actual sizes
                                                   int *__restrict__ NIterations,
                                                   const int *__restrict__ ovAtomTreePaddedSize, //padded allocated sizes
                                                   const int *__restrict__ ovAtomTreeLock,       //tree locks
                                                   const real *__restrict__ global_atomic_gamma, //atomic gamma
                                                   const int *__restrict__ ovLevel, //this and below define tree
                                                   real *__restrict__ ovGamma1i,
                                                   const int *__restrict__ ovLastAtom,
                                                   const int *__restrict__ ovRootIndex,
                                                   const int *__restrict__ ovChildrenStartIndex,
                                                   const int *__restrict__ ovChildrenCount,
                                                   volatile int *__restrict__ ovProcessedFlag,
                                                   volatile int *__restrict__ ovOKtoProcessFlag,
                                                   volatile int *__restrict__ ovChildrenReported) {

    const unsigned int local_id = threadIdx.x;
    const unsigned int gsize = OV_WORK_GROUP_SIZE;
    __shared__ unsigned int nprocessed;

    unsigned int tree = blockIdx.x;
    while (tree < ntrees) {
        unsigned int tree_ptr = ovTreePointer[tree];
        unsigned int tree_size = ovAtomTreeSize[tree];

        //this is the number of translations of the calculations window
        //to cover the tree section
        //make sure padded tree size is multiple of OV_WORK_GROUP_SIZE
        const unsigned int nsections = ovAtomTreePaddedSize[tree] / gsize;
        //start at the top of the tree and iterate until the end of the tree is reached
        for (unsigned int isection = 0; isection < nsections; isection++) {
            unsigned int slot = tree_ptr + isection * OV_WORK_GROUP_SIZE + local_id; //the slot to work on
//TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads();
            int parent = ovRootIndex[slot];
            int atom = ovLastAtom[slot];

            //for(unsigned int iiter = 0; iiter < 2 ; iiter++){
            do {
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
                if (local_id == 0) nprocessed = 0;
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
                int processed = ovProcessedFlag[slot];
                int ok2process = ovOKtoProcessFlag[slot];
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
                bool letsgo = (parent >= 0 && processed == 0 && ok2process > 0 && atom >= 0);
                if (letsgo) {
                    //atomic_inc(&nprocessed);
                    atomicAdd(&nprocessed, 1);
                    real gamma1 = ovGamma1i[parent];
                    real gamma2 = global_atomic_gamma[atom];
                    ovGamma1i[slot] = gamma1 + gamma2;
                    //mark itself as processed and children as okay to process
                    ovProcessedFlag[slot] = 1;
                    ovOKtoProcessFlag[slot] = 0;
                    if (ovChildrenStartIndex[slot] >= 0 && ovChildrenCount[slot] > 0) {
                        for (int i = ovChildrenStartIndex[slot];
                             i < ovChildrenStartIndex[slot] + ovChildrenCount[slot]; i++) {
                            ovOKtoProcessFlag[i] = 1;
                        }
                    }
                }
//TODOLater: Global memory fence needed or syncthreads sufficient?
                __syncthreads();
            } while (nprocessed > 0);
//TODOLater: Global memory fence needed or syncthreads sufficient?
            __syncthreads();
        }

        // next tree
        tree += gridDim.x;
//TODOLater: Global memory fence needed or syncthreads sufficient?
        __syncthreads();
    }
}
