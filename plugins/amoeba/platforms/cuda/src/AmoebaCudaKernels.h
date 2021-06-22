#ifndef AMOEBA_OPENMM_CUDAKERNELS_H_
#define AMOEBA_OPENMM_CUDAKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMAmoeba                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs, Peter Eastman                                    *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "openmm/amoebaKernels.h"
#include "openmm/kernels.h"
#include "openmm/System.h"
#include "CudaContext.h"
#include "CudaNonbondedUtilities.h"
#include "CudaSort.h"
#include "AmoebaCommonKernels.h"
#include <cufft.h>

namespace OpenMM {

/**
 * This kernel is invoked by AmoebaMultipoleForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcAmoebaMultipoleForceKernel : public CommonCalcAmoebaMultipoleForceKernel {
public:
    CudaCalcAmoebaMultipoleForceKernel(const std::string& name, const Platform& platform, CudaContext& cu, const System& system) :
            CommonCalcAmoebaMultipoleForceKernel(name, platform, cu, system), hasInitializedFFT(false) {
    }
    ~CudaCalcAmoebaMultipoleForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the AmoebaMultipoleForce this kernel will be used for
     */
    void initialize(const System& system, const AmoebaMultipoleForce& force);
    /**
     * Compute the FFT.
     */
    void computeFFT(bool forward);
    /**
     * Get whether charge spreading should be done in fixed point.
     */
    bool useFixedPointChargeSpreading() const {
        return cc.getUseDoublePrecision();
    }
private:
    bool hasInitializedFFT;
    cufftHandle fft;
};

/**
 * This kernel is invoked by HippoNonbondedForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcHippoNonbondedForceKernel : public CommonCalcHippoNonbondedForceKernel {
public:
    CudaCalcHippoNonbondedForceKernel(const std::string& name, const Platform& platform, CudaContext& cu, const System& system) :
            CommonCalcHippoNonbondedForceKernel(name, platform, cu, system), sort(NULL), hasInitializedFFT(false) {
    }
    ~CudaCalcHippoNonbondedForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the HippoNonbondedForce this kernel will be used for
     */
    void initialize(const System& system, const HippoNonbondedForce& force);
    /**
     * Compute the FFT.
     */
    void computeFFT(bool forward, bool dispersion);
    /**
     * Get whether charge spreading should be done in fixed point.
     */
    bool useFixedPointChargeSpreading() const {
        return cc.getUseDoublePrecision();
    }
    /**
     * Sort the atom grid indices.
     */
    void sortGridIndex();
private:
    class SortTrait : public CudaSort::SortTrait {
        int getDataSize() const {return 8;}
        int getKeySize() const {return 4;}
        const char* getDataType() const {return "int2";}
        const char* getKeyType() const {return "int";}
        const char* getMinKey() const {return "(-2147483647-1)";}
        const char* getMaxKey() const {return "2147483647";}
        const char* getMaxValue() const {return "make_int2(2147483647, 2147483647)";}
        const char* getSortKey() const {return "value.y";}
    };
    bool hasInitializedFFT;
    CudaSort* sort;
    cufftHandle fftForward, fftBackward, dfftForward, dfftBackward;
};

/**
 * This kernel is invoked by GKCavitationForce to calculate the forces acting on the system and the energy of the system.
 */
    class CudaCalcGKCavitationForceKernel : public CalcGKCavitationForceKernel {
    public:
        CudaCalcGKCavitationForceKernel(std::string name, const OpenMM::Platform &platform, OpenMM::CudaContext &cu,
                                        const OpenMM::System &system);

        ~CudaCalcGKCavitationForceKernel();

        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param force      the GKCavitationForce this kernel will be used for
         */
        void initialize(const OpenMM::System &system, const AmoebaGKCavitationForce &force);

        /**
         * Execute the kernel to calculate the forces and/or energy.
         *
         * @param context        the context in which to execute this kernel
         * @param includeForces  true if forces should be calculated
         * @param includeEnergy  true if the energy should be calculated
         * @return the potential energy due to the force
         */
        double execute(OpenMM::ContextImpl &context, bool includeForces, bool includeEnergy);

        /**
         * Copy changed parameters over to a context.
         *
         * @param context    the context to copy parameters to
         * @param force      the GKCavitationForce to copy the parameters from
         */
        void copyParametersToContext(OpenMM::ContextImpl &context, const AmoebaGKCavitationForce &force);

        class CudaOverlapTree {
        public:
            CudaOverlapTree(void) {
                ovAtomTreePointer = NULL;
                ovAtomTreeSize = NULL;
                ovTreePointer = NULL;
                ovNumAtomsInTree = NULL;
                ovFirstAtom = NULL;
                NIterations = NULL;
                ovAtomTreePaddedSize = NULL;
                ovAtomTreeLock = NULL;
                ovLevel = NULL;
                ovG = NULL;
                ovVolume = NULL;
                ovVsp = NULL;
                ovVSfp = NULL;
                ovSelfVolume = NULL;
                ovVolEnergy = NULL;
                ovGamma1i = NULL;
                ovDV1 = NULL;
                ovDV2 = NULL;
                ovPF = NULL;
                ovLastAtom = NULL;
                ovRootIndex = NULL;
                ovChildrenStartIndex = NULL;
                ovChildrenCount = NULL;
                ovChildrenCountTop = NULL;
                ovChildrenCountBottom = NULL;
                ovProcessedFlag = NULL;
                ovOKtoProcessFlag = NULL;
                ovChildrenReported = NULL;
                ovAtomBuffer = NULL;
                selfVolumeBuffer_long = NULL;
                selfVolumeBuffer = NULL;
                AccumulationBuffer1_long = NULL;
                AccumulationBuffer1_real = NULL;
                AccumulationBuffer2_long = NULL;
                AccumulationBuffer2_real = NULL;
                gradBuffers_long = NULL;
                temp_buffer_size = -1;
                gvol_buffer_temp = NULL;
                tree_pos_buffer_temp = NULL;
                i_buffer_temp = NULL;
                atomj_buffer_temp = NULL;
                has_saved_noverlaps = false;
                tree_size_boost = 2;//6;//debug 2 is default
                hasExceededTempBuffer = false;
            };

            ~CudaOverlapTree(void) {
                delete ovAtomTreePointer;
                delete ovAtomTreeSize;
                delete ovTreePointer;
                delete ovNumAtomsInTree;
                delete ovFirstAtom;
                delete NIterations;
                delete ovAtomTreePaddedSize;
                delete ovAtomTreeLock;
                delete ovLevel;
                delete ovG;
                delete ovVolume;
                delete ovVsp;
                delete ovVSfp;
                delete ovSelfVolume;
                delete ovVolEnergy;
                delete ovGamma1i;
                delete ovDV1;
                delete ovDV2;
                delete ovPF;
                delete ovLastAtom;
                delete ovRootIndex;
                delete ovChildrenStartIndex;
                delete ovChildrenCount;
                delete ovChildrenCountTop;
                delete ovChildrenCountBottom;
                delete ovProcessedFlag;
                delete ovOKtoProcessFlag;
                delete ovChildrenReported;
                delete ovAtomBuffer;
                delete selfVolumeBuffer_long;
                delete selfVolumeBuffer;
                delete AccumulationBuffer1_long;
                delete AccumulationBuffer1_real;
                delete AccumulationBuffer2_long;
                delete AccumulationBuffer2_real;
                delete gradBuffers_long;
                delete gvol_buffer_temp;
                delete tree_pos_buffer_temp;
                delete i_buffer_temp;
                delete atomj_buffer_temp;
            };

            //initializes tree sections and sizes with number of atoms and number of overlaps
            void init_tree_size(int num_atoms, int padded_num_atoms, int num_compute_units, int pad_modulo,
                                std::vector<int> &noverlaps_current);

            //resizes tree buffers
            void resize_tree_buffers(OpenMM::CudaContext &cu, int ov_work_group_size);

            //copies the tree framework to Cuda device memory
            int copy_tree_to_device(void);

            // host variables and buffers
            int num_atoms;
            int padded_num_atoms;
            int total_atoms_in_tree;
            int total_tree_size;
            int num_sections;
            std::vector<int> tree_size;
            std::vector<int> padded_tree_size;
            std::vector<int> atom_tree_pointer; //pointers to 1-body atom slots
            std::vector<int> tree_pointer;      //pointers to tree sections
            std::vector<int> natoms_in_tree;    //no. atoms in each tree section
            std::vector<int> first_atom;        //the first atom in each tree section

            /* overlap tree buffers on Device */
            OpenMM::CudaArray *ovAtomTreePointer;
            OpenMM::CudaArray *ovAtomTreeSize;
            OpenMM::CudaArray *ovTreePointer;
            OpenMM::CudaArray *ovNumAtomsInTree;
            OpenMM::CudaArray *ovFirstAtom;
            OpenMM::CudaArray *NIterations;
            OpenMM::CudaArray *ovAtomTreePaddedSize;
            OpenMM::CudaArray *ovAtomTreeLock;
            OpenMM::CudaArray *ovLevel;
            OpenMM::CudaArray *ovG; // real4: Gaussian position + exponent
            OpenMM::CudaArray *ovVolume;
            OpenMM::CudaArray *ovVsp;
            OpenMM::CudaArray *ovVSfp;
            OpenMM::CudaArray *ovSelfVolume;
            OpenMM::CudaArray *ovVolEnergy;
            OpenMM::CudaArray *ovGamma1i;
            /* volume derivatives */
            OpenMM::CudaArray *ovDV1; // real4: dV12/dr1 + dV12/dV1 for each overlap
            OpenMM::CudaArray *ovDV2; // volume gradient accumulator
            OpenMM::CudaArray *ovPF;  //(P) and (F) aux variables

            OpenMM::CudaArray *ovLastAtom;
            OpenMM::CudaArray *ovRootIndex;
            OpenMM::CudaArray *ovChildrenStartIndex;
            OpenMM::CudaArray *ovChildrenCount;
            OpenMM::CudaArray *ovChildrenCountTop;
            OpenMM::CudaArray *ovChildrenCountBottom;
            OpenMM::CudaArray *ovProcessedFlag;
            OpenMM::CudaArray *ovOKtoProcessFlag;
            OpenMM::CudaArray *ovChildrenReported;

            OpenMM::CudaArray *ovAtomBuffer;
            OpenMM::CudaArray *selfVolumeBuffer_long;
            OpenMM::CudaArray *selfVolumeBuffer;
            OpenMM::CudaArray *AccumulationBuffer1_long;
            OpenMM::CudaArray *AccumulationBuffer1_real;
            OpenMM::CudaArray *AccumulationBuffer2_long;
            OpenMM::CudaArray *AccumulationBuffer2_real;
            OpenMM::CudaArray *gradBuffers_long;

            int temp_buffer_size;
            OpenMM::CudaArray *gvol_buffer_temp;
            OpenMM::CudaArray *tree_pos_buffer_temp;
            OpenMM::CudaArray *i_buffer_temp;
            OpenMM::CudaArray *atomj_buffer_temp;

            double tree_size_boost;
            int has_saved_noverlaps;
            std::vector<int> saved_noverlaps;

            bool hasExceededTempBuffer;
        };//class CudaOverlapTree


    private:
        const AmoebaGKCavitationForce *gvol_force;

        int numParticles;
        unsigned int version;
        bool useCutoff;
        bool usePeriodic;
        bool useExclusions;
        double cutoffDistance;
        double roffset;
        float common_gamma;
        int maxTiles;
        bool hasInitializedKernels;
        bool hasCreatedKernels;
        OpenMM::CudaContext &cu;
        const OpenMM::System &system;
        int ov_work_group_size; //thread group size
        int num_compute_units;

        CudaOverlapTree *gtree;   //tree of atomic overlaps
        OpenMM::CudaArray *radiusParam1;
        OpenMM::CudaArray *radiusParam2;
        OpenMM::CudaArray *gammaParam1;
        OpenMM::CudaArray *gammaParam2;
        OpenMM::CudaArray *ishydrogenParam;

        //C++ vectors corresponding to parameter buffers above
        std::vector<float> radiusVector1; //enlarged radii
        std::vector<float> radiusVector2; //vdw radii
        std::vector<float> gammaVector1;  //gamma/radius_offset
        std::vector<float> gammaVector2;  //-gamma/radius_offset
        std::vector<int> ishydrogenVector;
        OpenMM::CudaArray *selfVolume; //vdw radii
        OpenMM::CudaArray *selfVolumeLargeR; //large radii
        OpenMM::CudaArray *Semaphor;
        OpenMM::CudaArray *grad;

        CUfunction resetBufferKernel;
        CUfunction resetOvCountKernel;
        CUfunction resetTree;
        CUfunction resetSelfVolumesKernel;
        CUfunction InitOverlapTreeKernel_1body_1;
        CUfunction InitOverlapTreeKernel_1body_2;
        CUfunction InitOverlapTreeCountKernel;
        CUfunction reduceovCountBufferKernel;
        CUfunction InitOverlapTreeKernel;
        CUfunction ComputeOverlapTreeKernel;
        CUfunction ComputeOverlapTree_1passKernel;
        CUfunction computeSelfVolumesKernel;
        CUfunction reduceSelfVolumesKernel_tree;
        CUfunction reduceSelfVolumesKernel_buffer;
        CUfunction updateSelfVolumesForcesKernel;
        CUfunction resetTreeKernel;
        CUfunction SortOverlapTree2bodyKernel;
        CUfunction resetComputeOverlapTreeKernel;
        CUfunction ResetRescanOverlapTreeKernel;
        CUfunction InitRescanOverlapTreeKernel;
        CUfunction RescanOverlapTreeKernel;
        CUfunction RescanOverlapTreeGammasKernel_W;
        CUfunction InitOverlapTreeGammasKernel_1body_W;

        /* Gaussian atomic parameters */
        std::vector<float> gaussian_exponent;
        std::vector<float> gaussian_volume;
        OpenMM::CudaArray *GaussianExponent;
        OpenMM::CudaArray *GaussianVolume;
        OpenMM::CudaArray *GaussianExponentLargeR;
        OpenMM::CudaArray *GaussianVolumeLargeR;

        /* gamma parameters */
        std::vector<float> atomic_gamma;
        OpenMM::CudaArray *AtomicGamma;
        std::vector<int> atom_ishydrogen;

        int niterations;

        void executeInitKernels(OpenMM::ContextImpl &context, bool includeForces, bool includeEnergy);

        double executeGVolSA(OpenMM::ContextImpl &context, bool includeForces, bool includeEnergy);

        //TODO: Panic Button?
        //flag to give up
        OpenMM::CudaArray *PanicButton;
        std::vector<int> panic_button;
        int *pinnedPanicButtonMemory;
        CUevent downloadPanicButtonEvent;
    };

} // namespace OpenMM

#endif /*AMOEBA_OPENMM_CUDAKERNELS_H*/
