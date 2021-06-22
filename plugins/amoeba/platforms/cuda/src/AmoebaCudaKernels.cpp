/* -------------------------------------------------------------------------- *
 *                               OpenMMAmoeba                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2020 Stanford University and the Authors.      *
 * Authors: Peter Eastman, Mark Friedrichs                                    *
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

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "AmoebaCudaKernels.h"
#include "CudaAmoebaKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/AmoebaGeneralizedKirkwoodForceImpl.h"
#include "openmm/internal/AmoebaMultipoleForceImpl.h"
#include "openmm/internal/AmoebaWcaDispersionForceImpl.h"
#include "openmm/internal/AmoebaTorsionTorsionForceImpl.h"
#include "openmm/internal/AmoebaVdwForceImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "CudaBondedUtilities.h"
#include "CudaFFT3D.h"
#include "CudaForceInfo.h"
#include "CudaKernelSources.h"
#include "SimTKOpenMMRealType.h"
#include "jama_lu.h"
#include "gaussvol.h"

#include <algorithm>
#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result, prefix) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<prefix<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

static void setPeriodicBoxArgs(ComputeContext& cc, ComputeKernel kernel, int index) {
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    if (cc.getUseDoublePrecision()) {
        kernel->setArg(index++, mm_double4(a[0], b[1], c[2], 0.0));
        kernel->setArg(index++, mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0));
        kernel->setArg(index++, mm_double4(a[0], a[1], a[2], 0.0));
        kernel->setArg(index++, mm_double4(b[0], b[1], b[2], 0.0));
        kernel->setArg(index, mm_double4(c[0], c[1], c[2], 0.0));
    }
    else {
        kernel->setArg(index++, mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f));
        kernel->setArg(index, mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f));
    }
}

/* -------------------------------------------------------------------------- *
 *                             AmoebaMultipole                                *
 * -------------------------------------------------------------------------- */

CudaCalcAmoebaMultipoleForceKernel::~CudaCalcAmoebaMultipoleForceKernel() {
    cc.setAsCurrent();
    if (hasInitializedFFT)
        cufftDestroy(fft);
}

void CudaCalcAmoebaMultipoleForceKernel::initialize(const System& system, const AmoebaMultipoleForce& force) {
    CommonCalcAmoebaMultipoleForceKernel::initialize(system, force);
    if (usePME) {
        cufftResult result = cufftPlan3d(&fft, gridSizeX, gridSizeY, gridSizeZ, cc.getUseDoublePrecision() ? CUFFT_Z2Z : CUFFT_C2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
        hasInitializedFFT = true;
    }
}

void CudaCalcAmoebaMultipoleForceKernel::computeFFT(bool forward) {
    CudaArray& grid1 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid1);
    CudaArray& grid2 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid2);
    if (forward) {
        if (cc.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) grid1.getDevicePointer(), (double2*) grid2.getDevicePointer(), CUFFT_FORWARD);
        else
            cufftExecC2C(fft, (float2*) grid1.getDevicePointer(), (float2*) grid2.getDevicePointer(), CUFFT_FORWARD);
    }
    else {
        if (cc.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) grid2.getDevicePointer(), (double2*) grid1.getDevicePointer(), CUFFT_INVERSE);
        else
            cufftExecC2C(fft, (float2*) grid2.getDevicePointer(), (float2*) grid1.getDevicePointer(), CUFFT_INVERSE);
    }
}

/* -------------------------------------------------------------------------- *
 *                           HippoNonbondedForce                              *
 * -------------------------------------------------------------------------- */

CudaCalcHippoNonbondedForceKernel::~CudaCalcHippoNonbondedForceKernel() {
    cc.setAsCurrent();
    if (sort != NULL)
        delete sort;
    if (hasInitializedFFT) {
        cufftDestroy(fftForward);
        cufftDestroy(fftBackward);
        cufftDestroy(dfftForward);
        cufftDestroy(dfftBackward);
    }
}

void CudaCalcHippoNonbondedForceKernel::initialize(const System& system, const HippoNonbondedForce& force) {
    CommonCalcHippoNonbondedForceKernel::initialize(system, force);
    if (usePME) {
        CudaContext& cu = dynamic_cast<CudaContext&>(cc);
        sort = new CudaSort(cu, new SortTrait(), cc.getNumAtoms());
        cufftResult result = cufftPlan3d(&fftForward, gridSizeX, gridSizeY, gridSizeZ, cc.getUseDoublePrecision() ? CUFFT_D2Z : CUFFT_R2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
        result = cufftPlan3d(&fftBackward, gridSizeX, gridSizeY, gridSizeZ, cc.getUseDoublePrecision() ? CUFFT_Z2D : CUFFT_C2R);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
        result = cufftPlan3d(&dfftForward, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, cc.getUseDoublePrecision() ? CUFFT_D2Z : CUFFT_R2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
        result = cufftPlan3d(&dfftBackward, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, cc.getUseDoublePrecision() ? CUFFT_Z2D : CUFFT_C2R);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
        hasInitializedFFT = true;
    }
}

void CudaCalcHippoNonbondedForceKernel::computeFFT(bool forward, bool dispersion) {
    CudaArray& grid1 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid1);
    CudaArray& grid2 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid2);
    if (forward) {
        cufftHandle fft = dispersion ? dfftForward : fftForward;
        if (cc.getUseDoublePrecision())
            cufftExecD2Z(fft, (double*) grid1.getDevicePointer(), (double2*) grid2.getDevicePointer());
        else
            cufftExecR2C(fft, (float*) grid1.getDevicePointer(), (float2*) grid2.getDevicePointer());
    }
    else {
        cufftHandle fft = dispersion ? dfftBackward : fftBackward;
        if (cc.getUseDoublePrecision())
            cufftExecZ2D(fft, (double2*) grid2.getDevicePointer(), (double*) grid1.getDevicePointer());
        else
            cufftExecC2R(fft, (float2*) grid2.getDevicePointer(), (float*) grid1.getDevicePointer());
    }
}

void CudaCalcHippoNonbondedForceKernel::sortGridIndex() {
    sort->sort(dynamic_cast<CudaContext&>(cc).unwrap(pmeAtomGridIndex));
}

/**
 * AmoebaGKCavitationForce Cuda Kernel
 */
CudaCalcGKCavitationForceKernel::CudaCalcGKCavitationForceKernel(std::string name, const OpenMM::Platform &platform, OpenMM::CudaContext &cu, const OpenMM::System &system) : CalcGKCavitationForceKernel(name, platform), cu(cu), system(system) {
    hasCreatedKernels = false;
    hasInitializedKernels = false;
    selfVolume = NULL;
    selfVolumeLargeR = NULL;
    Semaphor = NULL;
    GaussianExponent = NULL;
    GaussianVolume = NULL;
    GaussianExponentLargeR = NULL;
    GaussianVolumeLargeR = NULL;
    AtomicGamma = NULL;
    grad = NULL;
    PanicButton = NULL;
    pinnedPanicButtonMemory = NULL;
}

CudaCalcGKCavitationForceKernel::~CudaCalcGKCavitationForceKernel() {
    if (gtree != NULL) delete gtree;
    if (pinnedPanicButtonMemory != NULL) cuMemFreeHost(pinnedPanicButtonMemory);
    cuEventDestroy(downloadPanicButtonEvent);
}

//version based on number of overlaps for each atom
void CudaCalcGKCavitationForceKernel::CudaOverlapTree::init_tree_size(int num_atoms,
                                                                      int padded_num_atoms,
                                                                      int num_compute_units,
                                                                      int pad_modulo,
                                                                      vector<int> &noverlaps_current) {
    this->num_atoms = num_atoms;
    this->padded_num_atoms = padded_num_atoms;
    total_tree_size = 0;
    tree_size.clear();
    tree_pointer.clear();
    padded_tree_size.clear();
    atom_tree_pointer.clear();
    natoms_in_tree.clear();
    first_atom.clear();

    //The tree may be reinitialized multiple times due to too many overlaps.
    //Remember the largest number of overlaps per atom because if it went over the max before it
    //is likely to happen again
    if (!has_saved_noverlaps) {
        saved_noverlaps.resize(num_atoms);
        for (int i = 0; i < num_atoms; i++) saved_noverlaps[i] = 0;
        has_saved_noverlaps = true;
    }
    vector<int> noverlaps(num_atoms);
    for (int i = 0; i < num_atoms; i++) {
        noverlaps[i] = (saved_noverlaps[i] > noverlaps_current[i]) ? saved_noverlaps[i] : noverlaps_current[i] + 1;
        //(the +1 above counts the 1-body overlap)
    }
    for (int i = 0; i < num_atoms; i++) saved_noverlaps[i] = noverlaps[i];

    //assigns atoms to compute units (tree sections) in such a way that each compute unit gets
    //approximately equal number of overlaps
    num_sections = num_compute_units;
    vector<int> noverlaps_sum(num_atoms + 1);//prefix sum of number of overlaps per atom
    noverlaps_sum[0] = 0;
    for (int i = 1; i <= num_atoms; i++) {
        noverlaps_sum[i] = noverlaps[i - 1] + noverlaps_sum[i - 1];
    }
    int n_overlaps_total = noverlaps_sum[num_atoms];

    int max_n_overlaps = 0;
    for (int i = 0; i < num_atoms; i++) {
        if (noverlaps[i] > max_n_overlaps) max_n_overlaps = noverlaps[i];
    }

    int n_overlaps_per_section;
    if (num_sections > 1) {
        n_overlaps_per_section = n_overlaps_total / (num_sections - 1);
    } else {
        n_overlaps_per_section = n_overlaps_total;
    }
    if (max_n_overlaps > n_overlaps_per_section) n_overlaps_per_section = max_n_overlaps;

    //assigns atoms to compute units
    vector<int> compute_unit_of_atom(num_atoms);
    total_atoms_in_tree = 0;
    natoms_in_tree.resize(num_sections);
    for (int section = 0; section < num_sections; section++) {
        natoms_in_tree[section] = 0;
    }
    for (int i = 0; i < num_atoms; i++) {
        int section = noverlaps_sum[i] / n_overlaps_per_section;
        compute_unit_of_atom[i] = section;
        natoms_in_tree[section] += 1;
        total_atoms_in_tree += 1;
    }

    // computes sizes of tree sections
    vector<int> section_size(num_sections);
    for (int section = 0; section < num_sections; section++) {
        section_size[section] = 0;
    }
    for (int i = 0; i < num_atoms; i++) {
        int section = compute_unit_of_atom[i];
        section_size[section] += noverlaps[i];
    }
    //double sizes and pad for extra buffer
    for (int section = 0; section < num_sections; section++) {
        int tsize = section_size[section] < 1 ? 1 : section_size[section];
        tsize *= tree_size_boost;
        int npadsize = pad_modulo * ((tsize + pad_modulo - 1) / pad_modulo);
        section_size[section] = npadsize;
    }

    // set tree pointers
    tree_pointer.resize(num_sections);
    int offset = 0;
    for (int section = 0; section < num_sections; section++) {
        tree_pointer[section] = offset;
        offset += section_size[section];
    }

    // set atom pointer in tree
    tree_size.resize(num_sections);
    padded_tree_size.resize(num_sections);
    atom_tree_pointer.resize(padded_num_atoms);
    first_atom.resize(num_sections);
    int atom_offset = 0;
    for (int section = 0; section < num_sections; section++) {
        tree_size[section] = 0;
        padded_tree_size[section] = section_size[section];
        first_atom[section] = atom_offset;
        for (int i = 0; i < natoms_in_tree[section]; i++) {
            int iat = atom_offset + i;
            int slot = tree_pointer[section] + i;
            if (iat < total_atoms_in_tree) {
                atom_tree_pointer[iat] = slot;
            }
        }
        total_tree_size += section_size[section];
        atom_offset += natoms_in_tree[section];
    }

}

void CudaCalcGKCavitationForceKernel::CudaOverlapTree::resize_tree_buffers(OpenMM::CudaContext &cu, int ov_work_group_size) {
    if (ovAtomTreePointer) delete ovAtomTreePointer;
    ovAtomTreePointer = CudaArray::create<int>(cu, padded_num_atoms, "ovAtomTreePointer");
    if (ovAtomTreeSize) delete ovAtomTreeSize;
    ovAtomTreeSize = CudaArray::create<int>(cu, num_sections, "ovAtomTreeSize");
    if (NIterations) delete NIterations;
    NIterations = CudaArray::create<int>(cu, num_sections, "NIterations");
    if (ovAtomTreePaddedSize) delete ovAtomTreePaddedSize;
    ovAtomTreePaddedSize = CudaArray::create<int>(cu, num_sections, "ovAtomTreePaddedSize");
    if (ovNumAtomsInTree) delete ovNumAtomsInTree;
    ovNumAtomsInTree = CudaArray::create<int>(cu, num_sections, "ovNumAtomsInTree");
    if (ovTreePointer) delete ovTreePointer;
    ovTreePointer = CudaArray::create<int>(cu, num_sections, "ovTreePointer");
    if (ovAtomTreeLock) delete ovAtomTreeLock;
    ovAtomTreeLock = CudaArray::create<int>(cu, num_sections, "ovAtomTreeLock");
    if (ovFirstAtom) delete ovFirstAtom;
    ovFirstAtom = CudaArray::create<int>(cu, num_sections, "ovFirstAtom");
    if (ovLevel) delete ovLevel;
    ovLevel = CudaArray::create<int>(cu, total_tree_size, "ovLevel");
    if (ovG) delete ovG;
    ovG = CudaArray::create<float4>(cu, total_tree_size, "ovG"); //gaussian position + exponent
    if (ovVolume) delete ovVolume;
    ovVolume = CudaArray::create<float>(cu, total_tree_size, "ovVolume");
    if (ovVsp) delete ovVsp;
    ovVsp = CudaArray::create<float>(cu, total_tree_size, "ovVsp");
    if (ovVSfp) delete ovVSfp;
    ovVSfp = CudaArray::create<float>(cu, total_tree_size, "ovVSfp");
    if (ovSelfVolume) delete ovSelfVolume;
    ovSelfVolume = CudaArray::create<float>(cu, total_tree_size, "ovSelfVolume");
    if (ovVolEnergy) delete ovVolEnergy;
    ovVolEnergy = CudaArray::create<float>(cu, total_tree_size, "ovVolEnergy");
    if (ovGamma1i) delete ovGamma1i;
    ovGamma1i = CudaArray::create<float>(cu, total_tree_size, "ovGamma1i");
    if (ovDV1) delete ovDV1;
    ovDV1 = CudaArray::create<float4>(cu, total_tree_size, "ovDV1"); //dV12/dr1 + dV12/dV1 for each overlap
    if (ovDV2) delete ovDV2;
    ovDV2 = CudaArray::create<float4>(cu, total_tree_size, "ovDV2"); //volume gradient accumulator
    if (ovPF) delete ovPF;
    ovPF = CudaArray::create<float4>(cu, total_tree_size, "ovPF"); //(P) and (F) auxiliary variables
    if (ovLastAtom) delete ovLastAtom;
    ovLastAtom = CudaArray::create<int>(cu, total_tree_size, "ovLastAtom");
    if (ovRootIndex) delete ovRootIndex;
    ovRootIndex = CudaArray::create<int>(cu, total_tree_size, "ovRootIndex");
    if (ovChildrenStartIndex) delete ovChildrenStartIndex;
    ovChildrenStartIndex = CudaArray::create<int>(cu, total_tree_size, "ovChildrenStartIndex");
    if (ovChildrenCount) delete ovChildrenCount;
    ovChildrenCount = CudaArray::create<int>(cu, total_tree_size, "ovChildrenCount");
    if (ovChildrenCountTop) delete ovChildrenCountTop;
    ovChildrenCountTop = CudaArray::create<int>(cu, total_tree_size, "ovChildrenCountTop");
    if (ovChildrenCountBottom) delete ovChildrenCountBottom;
    ovChildrenCountBottom = CudaArray::create<int>(cu, total_tree_size, "ovChildrenCountBottom");
    if (ovProcessedFlag) delete ovProcessedFlag;
    ovProcessedFlag = CudaArray::create<int>(cu, total_tree_size, "ovProcessedFlag");
    if (ovOKtoProcessFlag) delete ovOKtoProcessFlag;
    ovOKtoProcessFlag = CudaArray::create<int>(cu, total_tree_size, "ovOKtoProcessFlag");
    if (ovChildrenReported) delete ovChildrenReported;
    ovChildrenReported = CudaArray::create<int>(cu, total_tree_size, "ovChildrenReported");


    // atomic reduction buffers, one for each tree section
    // used only if long int atomics are not available
    //   ovAtomBuffer holds volume energy derivatives (in xyz)
    if (ovAtomBuffer) delete ovAtomBuffer;
    ovAtomBuffer = CudaArray::create<float4>(cu, padded_num_atoms * num_sections, "ovAtomBuffer");

    //regular and "long" versions of selfVolume accumulation buffer (the latter updated using atomics)
    if (selfVolumeBuffer) delete selfVolumeBuffer;
    selfVolumeBuffer = CudaArray::create<float>(cu, padded_num_atoms * num_sections, "selfVolumeBuffer");
    if (selfVolumeBuffer_long) delete selfVolumeBuffer_long;
    selfVolumeBuffer_long = CudaArray::create<long>(cu, padded_num_atoms, "selfVolumeBuffer_long");

    //traditional and "long" versions of general accumulation buffers
    if (!AccumulationBuffer1_real) delete AccumulationBuffer1_real;
    AccumulationBuffer1_real = CudaArray::create<float>(cu, padded_num_atoms * num_sections,
                                                        "AccumulationBuffer1_real");
    if (!AccumulationBuffer1_long) delete AccumulationBuffer1_long;
    AccumulationBuffer1_long = CudaArray::create<long>(cu, padded_num_atoms, "AccumulationBuffer1_long");
    if (!AccumulationBuffer2_real) delete AccumulationBuffer2_real;
    AccumulationBuffer2_real = CudaArray::create<float>(cu, padded_num_atoms * num_sections,
                                                        "AccumulationBuffer2_real");
    if (!AccumulationBuffer2_long) delete AccumulationBuffer2_long;
    AccumulationBuffer2_long = CudaArray::create<long>(cu, padded_num_atoms, "AccumulationBuffer2_long");

    if (!gradBuffers_long) delete gradBuffers_long;
    gradBuffers_long = CudaArray::create<long>(cu, 4 * padded_num_atoms, "gradBuffers_long");

    //temp buffers to cache intermediate data in overlap tree construction (3-body and up)
    if (temp_buffer_size <= 0) {//first time
        int smax = 64; // this is n*(n-1)/2 where n is the max number of neighbors per overlap
        temp_buffer_size = ov_work_group_size * num_sections * smax;//first time
    }
    if (hasExceededTempBuffer) {//increase if needed
        temp_buffer_size = 2 * temp_buffer_size;
        hasExceededTempBuffer = false;
    }
    if (gvol_buffer_temp) delete gvol_buffer_temp;
    gvol_buffer_temp = CudaArray::create<float>(cu, temp_buffer_size, "gvol_buffer_temp");
    if (tree_pos_buffer_temp) delete tree_pos_buffer_temp;
    tree_pos_buffer_temp = CudaArray::create<unsigned int>(cu, temp_buffer_size, "tree_pos_buffer_temp");
    if (i_buffer_temp) delete i_buffer_temp;
    i_buffer_temp = CudaArray::create<int>(cu, temp_buffer_size, "i_buffer_temp");
    if (atomj_buffer_temp) delete atomj_buffer_temp;
    atomj_buffer_temp = CudaArray::create<int>(cu, temp_buffer_size, "atomj_buffer_temp");
}

int CudaCalcGKCavitationForceKernel::CudaOverlapTree::copy_tree_to_device(void) {

    vector<int> nn(padded_num_atoms);
    vector<int> ns(num_sections);

    for (int i = 0; i < padded_num_atoms; i++) {
        nn[i] = (int) atom_tree_pointer[i];
    }
    ovAtomTreePointer->upload(nn);

    for (int i = 0; i < num_sections; i++) {
        ns[i] = (int) tree_pointer[i];
    }
    ovTreePointer->upload(ns);

    for (int i = 0; i < num_sections; i++) {
        ns[i] = (int) tree_size[i];
    }
    ovAtomTreeSize->upload(ns);

    for (int i = 0; i < num_sections; i++) {
        ns[i] = (int) padded_tree_size[i];
    }
    ovAtomTreePaddedSize->upload(ns);

    for (int i = 0; i < num_sections; i++) {
        ns[i] = (int) natoms_in_tree[i];
    }
    ovNumAtomsInTree->upload(ns);

    for (int i = 0; i < num_sections; i++) {
        ns[i] = (int) first_atom[i];
    }
    ovFirstAtom->upload(ns);

    return 1;
}

void CudaCalcGKCavitationForceKernel::initialize(const System &system, const AmoebaGKCavitationForce &force) {
    roffset = GKCAV_RADIUS_INCREMENT;

    //we do not support multiple contexts(?), is it the same as multiple devices?
    if (cu.getPlatformData().contexts.size() > 1)
        throw OpenMMException("GKCavitationForce does not support using multiple contexts");

    CudaNonbondedUtilities &nb = cu.getNonbondedUtilities();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));

    numParticles = cu.getNumAtoms();//force.getNumParticles();
    if (numParticles == 0)
        return;
    radiusParam1 = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "radiusParam1");
    radiusParam2 = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "radiusParam2");
    gammaParam1 = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "gammaParam1");
    gammaParam2 = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "gammaParam2");
    ishydrogenParam = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(int), "ishydrogenParam");

    // this the accumulation buffer for overlap atom-level data (self-volumes, etc.)
    // note that each thread gets a separate buffer of size Natoms (rather than each thread block as in the
    // non-bonded algorithm). This may limits the max number of atoms.

    //cu.addAutoclearBuffer(*ovAtomBuffer);

    radiusVector1.resize(cu.getPaddedNumAtoms());
    radiusVector2.resize(cu.getPaddedNumAtoms());
    gammaVector1.resize(cu.getPaddedNumAtoms());
    gammaVector2.resize(cu.getPaddedNumAtoms());
    ishydrogenVector.resize(cu.getPaddedNumAtoms());
    atom_ishydrogen.resize(cu.getPaddedNumAtoms());
    common_gamma = -1;
    for (int i = 0; i < numParticles; i++) {
        double radius, gamma;
        bool ishydrogen;
        force.getParticleParameters(i, radius, gamma, ishydrogen);
        radiusVector1[i] = (float) radius + roffset;
        radiusVector2[i] = (float) radius;

        atom_ishydrogen[i] = ishydrogen ? 1 : 0;
        ishydrogenVector[i] = ishydrogen ? 1 : 0;

        // for surface-area energy use gamma/radius_offset
        // gamma = 1 for self volume calculation.
        double g = ishydrogen ? 0 : gamma / roffset;
        gammaVector1[i] = (float) g;
        gammaVector2[i] = (float) -g;

        //make sure that all gamma's are the same
        if (common_gamma < 0 && !ishydrogen) {
            common_gamma = gamma; //first occurrence of a non-zero gamma
        } else {
            if (!ishydrogen && pow(common_gamma - gamma, 2) > 1.e-6f) {
                throw OpenMMException("initialize(): GKCavitation does not support multiple gamma values.");
            }
        }

    }
    radiusParam1->upload(radiusVector1);
    radiusParam2->upload(radiusVector2);
    gammaParam1->upload(gammaVector1);
    gammaParam2->upload(gammaVector2);
    ishydrogenParam->upload(ishydrogenVector);
    useCutoff = (force.getNonbondedMethod() != AmoebaGKCavitationForce::NoCutoff);
    usePeriodic = (force.getNonbondedMethod() != AmoebaGKCavitationForce::NoCutoff &&
                   force.getNonbondedMethod() != AmoebaGKCavitationForce::CutoffNonPeriodic);
    useExclusions = false;
    cutoffDistance = force.getCutoffDistance();
    gtree = new CudaOverlapTree;//instance of atomic overlap tree
    CHECK_RESULT(cuEventCreate(&downloadPanicButtonEvent, 0), "Error creating event for GK cavitation force");
    CHECK_RESULT(cuMemHostAlloc((void**) &pinnedPanicButtonMemory, 2 * sizeof(int), CU_MEMHOSTALLOC_PORTABLE), "Error allocating PanicButton pinned buffer");
    gvol_force = &force;
    niterations = 0;
    hasInitializedKernels = false;
    hasCreatedKernels = false;
}

double CudaCalcGKCavitationForceKernel::execute(ContextImpl &context, bool includeForces, bool includeEnergy) {
    double energy = 0.0;
    if (!hasCreatedKernels || !hasInitializedKernels) {
        executeInitKernels(context, includeForces, includeEnergy);
        hasInitializedKernels = true;
        hasCreatedKernels = true;
    }
    energy = executeGVolSA(context, includeForces, includeEnergy);
    return 0.0;
}

void CudaCalcGKCavitationForceKernel::executeInitKernels(ContextImpl &context, bool includeForces, bool includeEnergy) {
    CudaNonbondedUtilities &nb = cu.getNonbondedUtilities();

    maxTiles = (nb.getUseCutoff() ? nb.getInteractingTiles().getSize() : 0);

    //run CPU version once to estimate sizes
    {
        GaussVol *gvol;
        std::vector<RealVec> positions;
        std::vector<int> ishydrogen;
        std::vector<RealOpenMM> radii;
        std::vector<RealOpenMM> gammas;
        //outputs
        RealOpenMM volume, vol_energy;
        std::vector<RealOpenMM> free_volume, self_volume;
        std::vector<RealVec> vol_force;
        std::vector<RealOpenMM> vol_dv;
        int numParticles = cu.getNumAtoms();
        //input lists
        positions.resize(numParticles);
        radii.resize(numParticles);
        gammas.resize(numParticles);
        ishydrogen.resize(numParticles);
        //output lists
        free_volume.resize(numParticles);
        self_volume.resize(numParticles);
        vol_force.resize(numParticles);
        vol_dv.resize(numParticles);

        for (int i = 0; i < numParticles; i++) {
            double r, g;
            bool h;
            gvol_force->getParticleParameters(i, r, g, h);
            radii[i] = r + roffset;
            gammas[i] = g / roffset; //energy_density_param;
            if (h) gammas[i] = 0.0;
            ishydrogen[i] = h ? 1 : 0;
        }
        gvol = new GaussVol(numParticles, ishydrogen);

        if (cu.getUseDoublePrecision()){
            vector<double4> posq;
            cu.getPosq().download(posq);
            for (int i = 0; i < numParticles; i++) {
                positions[i] = RealVec((RealOpenMM) posq[i].x, (RealOpenMM) posq[i].y, (RealOpenMM) posq[i].z);
            }
        }
        else{
            vector<float4> posq;
            cu.getPosq().download(posq);
            for (int i = 0; i < numParticles; i++) {
                positions[i] = RealVec((RealOpenMM) posq[i].x, (RealOpenMM) posq[i].y, (RealOpenMM) posq[i].z);
            }
        }

        vector<RealOpenMM> volumes(numParticles);
        for (int i = 0; i < numParticles; i++) {
            volumes[i] = 4. * M_PI * pow(radii[i], 3) / 3.;
        }

        gvol->setRadii(radii);
        gvol->setVolumes(volumes);
        gvol->setGammas(gammas);
        gvol->compute_tree(positions);
        //gvol->compute_volume(positions, volume, vol_energy, vol_force, vol_dv, free_volume, self_volume);
        vector<int> noverlaps(cu.getPaddedNumAtoms());
        for (int i = 0; i < cu.getPaddedNumAtoms(); i++) noverlaps[i] = 0;
        gvol->getstat(noverlaps);


        int nn = 0;
        for (int i = 0; i < noverlaps.size(); i++) {
            nn += noverlaps[i];
        }
        ov_work_group_size = nb.getForceThreadBlockSize();
        num_compute_units = nb.getNumForceThreadBlocks();

        //creates overlap tree
        int pad_modulo = ov_work_group_size;
        gtree->init_tree_size(cu.getNumAtoms(), cu.getPaddedNumAtoms(), num_compute_units, pad_modulo, noverlaps);
        //allocates or re-allocates tree buffers
        gtree->resize_tree_buffers(cu, ov_work_group_size);
        //copy overlap tree buffers to device
        gtree->copy_tree_to_device();

        delete gvol; //no longer needed

        //Sets up buffers
        //TODO: Panic Button?
        //sets up flag to detect when tree size is exceeded
        if (PanicButton) delete PanicButton;
        // pos 0 is general panic, pos 1 indicates execeeded temp buffer
        PanicButton = CudaArray::create<int>(cu, 2, "PanicButton");
        panic_button.resize(2);
        panic_button[0] = panic_button[1] = 0;    //init with zero
        PanicButton->upload(panic_button);

        // atom-level properties
        if (selfVolume) delete selfVolume;
        selfVolume = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "selfVolume");
        if (selfVolumeLargeR) delete selfVolumeLargeR;
        selfVolumeLargeR = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "selfVolumeLargeR");
        if (Semaphor) delete Semaphor;
        Semaphor = CudaArray::create<int>(cu, cu.getPaddedNumAtoms(), "Semaphor");
        vector<int> semaphor(cu.getPaddedNumAtoms());
        for (int i = 0; i < cu.getPaddedNumAtoms(); i++) semaphor[i] = 0;
        Semaphor->upload(semaphor);

        //atomic parameters
        if (GaussianExponent) delete GaussianExponent;
        GaussianExponent = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "GaussianExponent");
        if (GaussianVolume) delete GaussianVolume;
        GaussianVolume = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "GaussianVolume");
        if (GaussianExponentLargeR) delete GaussianExponentLargeR;
        GaussianExponentLargeR = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "GaussianExponentLargeR");
        if (GaussianVolumeLargeR) delete GaussianVolumeLargeR;
        GaussianVolumeLargeR = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "GaussianVolumeLargeR");
        if (AtomicGamma) delete AtomicGamma;
        AtomicGamma = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "AtomicGamma");
        if (grad) delete grad;
        grad = CudaArray::create<float4>(cu, cu.getPaddedNumAtoms(), "grad");

    }

    // Reset tree kernel compile
    {
        map<string, string> defines;
        defines["FORCE_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        defines["NUM_ATOMS_TREE"] = cu.intToString(gtree->total_atoms_in_tree);
        defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        defines["NUM_BLOCKS"] = cu.intToString(gtree->num_sections);
        defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);

        map<string, string> replacements;
        string file, kernel_name;
        CUmodule module;

        kernel_name = "resetTree";
        if (!hasCreatedKernels) {
            file = cu.replaceStrings(CudaAmoebaKernelSources::GVolResetTree, replacements);
            module = cu.createModule(file, defines);
            resetTreeKernel = cu.getKernel(module, kernel_name);
            // reset tree kernel
        }

        // reset buffer kernel
        kernel_name = "resetBuffer";
        if (!hasCreatedKernels) {
            resetBufferKernel= cu.getKernel(module, kernel_name);
        }


        // reset tree counters kernel
        kernel_name = "resetSelfVolumes";
        if (!hasCreatedKernels) {
            resetSelfVolumesKernel= cu.getKernel(module, kernel_name);
        }
    }

    //Tree construction compile
    {
        CUmodule module;
        string kernel_name;

        //pass 1
        map<string, string> pairValueDefines;
        pairValueDefines["FORCE_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        pairValueDefines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        pairValueDefines["NUM_ATOMS_TREE"] = cu.intToString(gtree->total_atoms_in_tree);
        pairValueDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        pairValueDefines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        pairValueDefines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        pairValueDefines["OV_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        pairValueDefines["SMALL_VOLUME"] = "1.e-4";
        pairValueDefines["MAX_ORDER"] = cu.intToString(MAX_ORDER);

        if (useCutoff)
            pairValueDefines["USE_CUTOFF"] = "1";
        if (usePeriodic)
            pairValueDefines["USE_PERIODIC"] = "1";
        pairValueDefines["USE_EXCLUSIONS"] = "1";
        pairValueDefines["CUTOFF"] = cu.doubleToString(cutoffDistance);
        pairValueDefines["CUTOFF_SQUARED"] = cu.doubleToString(cutoffDistance * cutoffDistance);
        int numContexts = cu.getPlatformData().contexts.size();
        int numExclusionTiles = nb.getExclusionTiles().getSize();
        pairValueDefines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
        int startExclusionIndex = cu.getContextIndex() * numExclusionTiles / numContexts;
        int endExclusionIndex = (cu.getContextIndex() + 1) * numExclusionTiles / numContexts;
        pairValueDefines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
        pairValueDefines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);


        map<string, string> replacements;

        replacements["KFC"] = cu.doubleToString((double) KFC);
        replacements["VOLMINA"] = cu.doubleToString((double) VOLMINA);
        replacements["VOLMINB"] = cu.doubleToString((double) VOLMINB);
        replacements["MIN_GVOL"] = cu.doubleToString((double) MIN_GVOL);

        replacements["ATOM_PARAMETER_DATA"] =
                "real4 g; \n"
                "real  v; \n"
                "real  gamma; \n"
                "int tree_pointer; \n";

        replacements["PARAMETER_ARGUMENTS"] = "";

        replacements["INIT_VARS"] = "";

        replacements["LOAD_ATOM1_PARAMETERS"] =
                "real a1 = global_gaussian_exponent[atom1]; \n"
                "real v1 = global_gaussian_volume[atom1];\n"
                "real gamma1 = global_atomic_gamma[atom1];\n";

        replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] =
                "localData[localAtomIndex].g.w = a1;\n"
                "localData[localAtomIndex].v = v1;\n"
                "localData[localAtomIndex].gamma = gamma1;\n";


        replacements["LOAD_ATOM2_PARAMETERS"] =
                "real a2 = localData[localAtom2Index].g.w;\n"
                "real v2 = localData[localAtom2Index].v;\n"
                "real gamma2 = localData[localAtom2Index].gamma;\n";

        replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] =
                "localData[localAtomIndex].g.w = global_gaussian_exponent[j];\n"
                "localData[localAtomIndex].v = global_gaussian_volume[j];\n"
                "localData[localAtomIndex].gamma = global_atomic_gamma[j];\n"
                "localData[localAtomIndex].ov_count = 0;\n";




        //tree locks were used in the 2-body tree construction kernel. no more
        replacements["ACQUIRE_TREE_LOCK"] = "";
        replacements["RELEASE_TREE_LOCK"] = "";

        replacements["COMPUTE_INTERACTION_COUNT"] =
                "       real a12 = a1 + a2; \n"
                "       real deltai = 1./a12; \n"
                "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
                "       real dfp = df/PI; \n"
                "       real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       if(gvol > VolMinA ){ \n" //VolMin0?
                "          atomicAdd((int *)&ovChildrenCount[parent_slot], 1); \n"
                "       } \n";

        replacements["COMPUTE_INTERACTION_2COUNT"] =
                "       real a12 = a1 + a2; \n"
                "       real deltai = 1./a12; \n"
                "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
                "       real dfp = df/PI; \n"
                "       real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       if(gvol > VolMinA ){ \n"
                "          ov_count += 1; \n"
                "       } \n";

        replacements["COMPUTE_INTERACTION_GVOLONLY"] =
                "       real a12 = a1 + a2; \n"
                "       real df = a1*a2/a12; \n"
                "       real ef = exp(-df*r2); \n"
                "       real dfp = df/PI; \n"
                "       real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n";

        replacements["COMPUTE_INTERACTION_OTHER"] =
                "         real a12 = a1 + a2; \n"
                "         real df = a1*a2/a12; \n"
                "         real dgvol = -2.0f*df*gvol; \n"
                "         real dgvolv = v1 > 0 ? gvol/v1 : 0; \n"
                "         //real4 c12 = (a1*posq1 + a2*posq2)/a12; \n"
                "       real4 c12 = make_real4((a1*posq1.x + a2*posq2.x)/a12, (a1*posq1.y + a2*posq2.y)/a12, (a1*posq1.z + a2*posq2.z)/a12, (a1*posq1.w + a2*posq2.w)/a12); \n"
                "         //switching function \n"
                "         real s = 0, sp = 0; \n"
                "         if(gvol > VolMinB ){ \n"
                "             s = 1.0f; \n"
                "             sp = 0.0f; \n"
                "         }else{ \n"
                "             real swd = 1.f/( VolMinB - VolMinA ); \n"
                "             real swu = (gvol - VolMinA)*swd; \n"
                "             real swu2 = swu*swu; \n"
                "             real swu3 = swu*swu2; \n"
                "             s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "             sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
                "         }\n"
                "         // switching function end \n"
                "         real sfp = sp*gvol + s; \n";


        replacements["COMPUTE_INTERACTION_STORE1"] =
                "       real a12 = a1 + a2; \n"
                "       real deltai = 1./a12; \n"
                "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
                "       real dfp = df/PI; \n"
                "       real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       if(gvol > VolMinA){\n"
                "         real dgvol = -2.0f*df*gvol; \n"
                "         real dgvolv = v1 > 0 ? gvol/v1 : 0; \n"
                "              //real4 c12 = deltai*(a1*posq1 + a2*posq2); \n"
                "         real4 c12 = make_real4(deltai*(a1*posq1.x + a2*posq2.x), deltai*(a1*posq1.y + a2*posq2.y), deltai*(a1*posq1.z + a2*posq2.z), deltai*(a1*posq1.w + a2*posq2.w)); \n"
                "         //switching function \n"
                "         real s = 0, sp = 0; \n"
                "         if(gvol > VolMinB ){ \n"
                "             s = 1.0f; \n"
                "             sp = 0.0f; \n"
                "         }else{ \n"
                "             real swd = 1.f/( VolMinB - VolMinA ); \n"
                "             real swu = (gvol - VolMinA)*swd; \n"
                "             real swu2 = swu*swu; \n"
                "             real swu3 = swu*swu2; \n"
                "             s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "             sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
                "         }\n"
                "         // switching function end \n"
                "         real sfp = sp*gvol + s; \n"
                "         /* at this point have:\n"
                "            1. gvol: overlap  between atom1 and atom2\n"
                "            2. a12: gaussian exponent of overlap\n"
                "            3. v12=gvol: volume of overlap\n"
                "            4. c12: gaussian center of overlap\n"
                "            These, together with atom2 (last_atom) are entered into the tree for atom 1 if\n"
                "            volume is large enough.\n"
                "        */\n"
                "        int endslot, children_count;\n"
                "        if(s*gvol > SMALL_VOLUME){ \n"
                "          //use top counter \n"
                "          children_count = atomicAdd(&ovChildrenCountTop[parent_slot], 1); \n"
                "          endslot = parent_children_start + children_count; \n"
                "        }else{ \n"
                "          //use bottom counter \n"
                "          children_count = atomicAdd(&ovChildrenCountBottom[parent_slot], 1); \n"
                "          endslot = parent_children_start + ovChildrenCount[parent_slot] - children_count - 1; \n"
                "        }\n"
                "        ovLevel[endslot] = 2; //two-body\n"
                "        ovVolume[endslot] = gvol;\n"
                "        ovVsp[endslot] = s; \n"
                "        ovVSfp[endslot] = sfp; \n"
                "        ovGamma1i[endslot] = gamma1 + gamma2;\n"
                "        ovLastAtom[endslot] = child_atom;\n"
                "        ovRootIndex[endslot] = parent_slot;\n"
                "        ovChildrenStartIndex[endslot] = -1;\n"
                "        ovChildrenCount[endslot] = 0;\n"
                "        //ovG[endslot] = (real4)(c12.xyz, a12);\n"
                "        //ovDV1[endslot] = (real4)(-delta.xyz*dgvol,dgvolv);\n"
                "    ovG[endslot] = make_real4(c12.x, c12.y, c12.z, a12);\n"
                "        ovDV1[endslot] = make_real4(-delta.x*dgvol, -delta.y*dgvol, -delta.z*dgvol, dgvolv);\n"
                "      }\n";


        replacements["COMPUTE_INTERACTION_STORE2"] =
                "       real a12 = a1 + a2; \n"
                "       real deltai = 1./a12; \n"
                "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
                "       real dfp = df/PI; \n"
                "       real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       if(gvol > VolMinA){\n"
                "         real dgvol = -2.0f*df*gvol; \n"
                "         real dgvolv = v1 > 0 ? gvol/v1 : 0; \n"
                "         //real4 c12 = deltai*(a1*posq1 + a2*posq2); \n"
                "     real4 c12 = make_real4(deltai*(a1*posq1.x + a2*posq2.x), deltai*(a1*posq1.y + a2*posq2.y), deltai*(a1*posq1.z + a2*posq2.z), deltai*(a1*posq1.w + a2*posq2.w)); \n"
                "         //switching function \n"
                "         real s = 0, sp = 0; \n"
                "         if(gvol > VolMinB ){ \n"
                "             s = 1.0f; \n"
                "             sp = 0.0f; \n"
                "         }else{ \n"
                "             real swd = 1.f/( VolMinB - VolMinA ); \n"
                "             real swu = (gvol - VolMinA)*swd; \n"
                "             real swu2 = swu*swu; \n"
                "             real swu3 = swu*swu2; \n"
                "             s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "             sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
                "         }\n"
                "         // switching function end \n"
                "         real sfp = sp*gvol + s; \n"
                "         /* at this point have:\n"
                "            1. gvol: overlap  between atom1 and atom2\n"
                "            2. a12: gaussian exponent of overlap\n"
                "            3. v12=gvol: volume of overlap\n"
                "            4. c12: gaussian center of overlap\n"
                "            These, together with atom2 (last_atom) are entered into the tree for atom 1 if\n"
                "            volume is large enough.\n"
                "        */\n"
                "        int endslot, children_count;\n"
                "        if(s*gvol > SMALL_VOLUME){ \n"
                "          //use top counter \n"
                "          children_count = ovChildrenCountTop[slot]++; \n"
                "          endslot = ovChildrenStartIndex[slot] + children_count; \n"
                "        }else{ \n"
                "          //use bottom counter \n"
                "          children_count = ovChildrenCountBottom[slot]++; \n"
                "          endslot = ovChildrenStartIndex[slot] + ovChildrenCount[slot] - children_count - 1; \n"
                "        }\n"
                "         ovLevel[endslot] = level + 1; //two-body\n"
                "         ovVolume[endslot] = gvol;\n"
                "         ovVsp[endslot] = s; \n"
                "         ovVSfp[endslot] = sfp; \n"
                "         ovGamma1i[endslot] = gamma1 + gamma2;\n"
                "         ovLastAtom[endslot] = atom2;\n"
                "         ovRootIndex[endslot] = slot;\n"
                "         ovChildrenStartIndex[endslot] = -1;\n"
                "         ovChildrenCount[endslot] = 0;\n"
                "         //ovG[endslot] = (real4)(c12.xyz, a12);\n"
                "         //ovDV1[endslot] = (real4)(-delta.xyz*dgvol,dgvolv);\n"
                "     ovG[endslot] = make_real4(c12.x, c12.y, c12.z, a12);\n"
                "         ovDV1[endslot] = make_real4(-delta.x*dgvol, -delta.y*dgvol, -delta.z*dgvol, dgvolv); \n"
                "         ovProcessedFlag[endslot] = 0;\n"
                "         ovOKtoProcessFlag[endslot] = 1;\n"
                "       }\n";


        replacements["COMPUTE_INTERACTION_RESCAN"] =
                "       real a12 = a1 + a2; \n"
                "       real deltai = 1./a12; \n"
                "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
                "       real dfp = df/PI; \n"
                "       real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       real dgvol = -2.0f*df*gvol; \n"
                "       real dgvolv = v1 > 0 ? gvol/v1 : 0; \n"
                "       //real4 c12 = deltai*(a1*posq1 + a2*posq2); \n"
                "       real4 c12 = make_real4(deltai*(a1*posq1.x + a2*posq2.x), deltai*(a1*posq1.y + a2*posq2.y), deltai*(a1*posq1.z + a2*posq2.z), deltai*(a1*posq1.w + a2*posq2.w)); \n"
                "       //switching function \n"
                "       real s = 0, sp = 0; \n"
                "       if(gvol > VolMinB ){ \n"
                "           s = 1.0f; \n"
                "           sp = 0.0f; \n"
                "       }else{ \n"
                "           real swd = 1.f/( VolMinB - VolMinA ); \n"
                "           real swu = (gvol - VolMinA)*swd; \n"
                "           real swu2 = swu*swu; \n"
                "           real swu3 = swu*swu2; \n"
                "           s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "           sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
                "       }\n"
                "       // switching function end \n"
                "       real sfp = sp*gvol + s; \n"
                "       ovVolume[slot] = gvol;\n"
                "       ovVsp[slot] = s; \n"
                "       ovVSfp[slot] = sfp; \n"
                "       //ovG[slot] = (real4)(c12.xyz, a12);\n"
                "       //ovDV1[slot] = (real4)(-delta.xyz*dgvol,dgvolv);\n"
                "       ovG[slot] = make_real4(c12.x, c12.y, c12.z, a12);\n"
                "       ovDV1[slot] = make_real4(-delta.x*dgvol, -delta.y*dgvol, -delta.z*dgvol, dgvolv); \n";


        int reset_tree_size;

        string InitOverlapTreeSrc;

        kernel_name = "InitOverlapTree_1body";//large radii
        if (!hasCreatedKernels) {
            InitOverlapTreeSrc = cu.replaceStrings(CudaAmoebaKernelSources::GVolOverlapTree, replacements);
            replacements["KERNEL_NAME"] = kernel_name;
            module = cu.createModule(InitOverlapTreeSrc, pairValueDefines);
            InitOverlapTreeKernel_1body_1 = cu.getKernel(module, kernel_name);
        }
        reset_tree_size = 1;

        if (!hasCreatedKernels) {
            module = cu.createModule(InitOverlapTreeSrc, pairValueDefines);
            InitOverlapTreeKernel_1body_2 = cu.getKernel(module, kernel_name);
        }
        reset_tree_size = 0;

        kernel_name = "InitOverlapTreeCount";
        replacements["KERNEL_NAME"] = kernel_name;

        if (!hasCreatedKernels) {
            InitOverlapTreeCountKernel = cu.getKernel(module, kernel_name);
        }


        if (!hasCreatedKernels) {
            kernel_name = "reduceovCountBuffer";
            replacements["KERNEL_NAME"] = kernel_name;
            reduceovCountBufferKernel = cu.getKernel(module, kernel_name);
        }


        if (!hasCreatedKernels) {
            kernel_name = "InitOverlapTree";
            replacements["KERNEL_NAME"] = kernel_name;
            InitOverlapTreeKernel = cu.getKernel(module, kernel_name);
        }

        if (!hasCreatedKernels) {
            kernel_name = "resetComputeOverlapTree";
            module = cu.createModule(InitOverlapTreeSrc, pairValueDefines);
            resetComputeOverlapTreeKernel = cu.getKernel(module, kernel_name);
        }


        //pass 2 (1 pass kernel)
        if (!hasCreatedKernels) {
            kernel_name = "ComputeOverlapTree_1pass";
            replacements["KERNEL_NAME"] = kernel_name;
            ComputeOverlapTree_1passKernel = cu.getKernel(module, kernel_name);
        }

        //2-body volumes sort kernel
        if(!hasCreatedKernels) {
            kernel_name = "SortOverlapTree2body";
            replacements["KERNEL_NAME"] = kernel_name;
            SortOverlapTree2bodyKernel = cu.getKernel(module, kernel_name);
        }

        //rescan kernels
        if(!hasCreatedKernels) {
            kernel_name = "ResetRescanOverlapTree";
            replacements["KERNEL_NAME"] = kernel_name;
            ResetRescanOverlapTreeKernel = cu.getKernel(module, kernel_name);
        }

        if (!hasCreatedKernels) {
            kernel_name = "InitRescanOverlapTree";
            replacements["KERNEL_NAME"] = kernel_name;
            InitRescanOverlapTreeKernel = cu.getKernel(module, kernel_name);
        }

        //propagates atomic parameters (radii, gammas, etc) from the top to the bottom
        //of the overlap tree, recomputes overlap volumes as it goes
        if (!hasCreatedKernels) {
            kernel_name = "RescanOverlapTree";
            replacements["KERNEL_NAME"] = kernel_name;
            RescanOverlapTreeKernel = cu.getKernel(module, kernel_name);
        }

        //seeds tree with van der Waals + GB gamma parameters
        if (!hasCreatedKernels) {
            kernel_name = "InitOverlapTreeGammas_1body";
            InitOverlapTreeGammasKernel_1body_W = cu.getKernel(module, kernel_name);
        }

        //Same as RescanOverlapTree above:
        //propagates van der Waals + GB gamma atomic parameters from the top to the bottom
        //of the overlap tree,
        //it *does not* recompute overlap volumes
        //  used to prep calculations of volume derivatives of van der Waals energy
        if (!hasCreatedKernels) {
            kernel_name = "RescanOverlapTreeGammas";
            replacements["KERNEL_NAME"] = kernel_name;
            RescanOverlapTreeGammasKernel_W = cu.getKernel(module, kernel_name);
        }
    }

    //Self volumes kernel compile
    {

        map<string, string> defines;
        defines["FORCE_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        defines["NUM_ATOMS_TREE"] = cu.intToString(gtree->total_atoms_in_tree);
        defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        defines["OV_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        map<string, string> replacements;
        CUmodule module;
        string kernel_name;
        string file;

        kernel_name = "computeSelfVolumes";
        if (!hasCreatedKernels) {
            file = cu.replaceStrings(CudaAmoebaKernelSources::GVolSelfVolume, replacements);
            defines["DO_SELF_VOLUMES"] = "1";
            module = cu.createModule(file, defines);
            //accumulates self volumes and volume energy function (and forces)
            //with the energy-per-unit-volume parameters (Gamma1i) currently loaded into tree
            computeSelfVolumesKernel = cu.getKernel(module, kernel_name);
        }
    }

    //Self volumes reduction kernel (pass 2) compile
    {
        map<string, string> defines;
        defines["FORCE_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        defines["NUM_ATOMS_TREE"] = cu.intToString(gtree->total_atoms_in_tree);
        defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        defines["NTILES_IN_BLOCK"] = "1";//cu.intToString(ov_work_group_size/CudaContext::TileSize);


        map<string, string> replacements;
        string kernel_name, file;
        CUmodule module;

        kernel_name = "reduceSelfVolumes_buffer";
        if (!hasCreatedKernels) {
            file = CudaAmoebaKernelSources::GVolReduceTree;
            module = cu.createModule(file, defines);
            reduceSelfVolumesKernel_buffer = cu.getKernel(module, kernel_name);
        }


        kernel_name = "updateSelfVolumesForces";
        if (!hasCreatedKernels) {
            updateSelfVolumesForcesKernel = cu.getKernel(module, kernel_name);
        }
    }
}

double CudaCalcGKCavitationForceKernel::executeGVolSA(ContextImpl &context, bool includeForces, bool includeEnergy) {
    CudaNonbondedUtilities &nb = cu.getNonbondedUtilities();
    niterations += 1;
    bool nb_reassign = false;
    if (useCutoff) {
        if (maxTiles < nb.getInteractingTiles().getSize()) {
            maxTiles = nb.getInteractingTiles().getSize();
            nb_reassign = true;
        }
    }

    unsigned int num_sections = gtree->num_sections;
    unsigned int paddedNumAtoms = cu.getPaddedNumAtoms();
    unsigned int numAtoms = cu.getNumAtoms();
    //------------------------------------------------------------------------------------------------------------
    // Tree construction (large radii)
    //
    //Execute resetTreeKernel
    {
        //here workgroups cycle through tree sections to reset the tree section
        void *resetTreeKernelArgs[] = {&num_sections,
                                       &gtree->ovTreePointer->getDevicePointer(),
                                       &gtree->ovAtomTreePointer->getDevicePointer(),
                                       &gtree->ovAtomTreeSize->getDevicePointer(),
                                       &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                       &gtree->ovLevel->getDevicePointer(),
                                       &gtree->ovVolume->getDevicePointer(),
                                       &gtree->ovVsp->getDevicePointer(),
                                       &gtree->ovVSfp->getDevicePointer(),
                                       &gtree->ovSelfVolume->getDevicePointer(),
                                       &gtree->ovVolEnergy->getDevicePointer(),
                                       &gtree->ovLastAtom->getDevicePointer(),
                                       &gtree->ovRootIndex->getDevicePointer(),
                                       &gtree->ovChildrenStartIndex->getDevicePointer(),
                                       &gtree->ovChildrenCount->getDevicePointer(),
                                       &gtree->ovDV1->getDevicePointer(),
                                       &gtree->ovDV2->getDevicePointer(),
                                       &gtree->ovProcessedFlag->getDevicePointer(),
                                       &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                       &gtree->ovChildrenReported->getDevicePointer(),
                                       &gtree->ovAtomTreeLock->getDevicePointer(),
                                       &gtree->NIterations->getDevicePointer()
        };
        cu.executeKernel(resetTreeKernel, resetTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute resetBufferKernel
    {
        // resets either ovAtomBuffer and long energy buffer
        void *resetBufferKernelArgs[] = {&paddedNumAtoms,
                                         &num_sections,
                                         &gtree->ovAtomBuffer->getDevicePointer(),
                                         &gtree->selfVolumeBuffer->getDevicePointer(),
                                         &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                         &gtree->gradBuffers_long->getDevicePointer()};
        cu.executeKernel(resetBufferKernel, resetBufferKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute InitOverlapTreeKernel_1body_1
    {
        //fills up tree with 1-body overlaps
        unsigned int reset_tree_size =1;
        void *InitOverlapTreeKernel_1body_1Args[] = {&paddedNumAtoms,
                                                     &num_sections,
                                                     &reset_tree_size,
                                                     &gtree->ovTreePointer->getDevicePointer(),
                                                     &gtree->ovNumAtomsInTree->getDevicePointer(),
                                                     &gtree->ovFirstAtom->getDevicePointer(),
                                                     &gtree->ovAtomTreeSize->getDevicePointer(),
                                                     &gtree->NIterations->getDevicePointer(),
                                                     &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                     &gtree->ovAtomTreePointer->getDevicePointer(),
                                                     &cu.getPosq().getDevicePointer(),
                                                     &radiusParam1->getDevicePointer(),
                                                     &gammaParam1->getDevicePointer(),
                                                     &ishydrogenParam->getDevicePointer(),
                                                     &GaussianExponent->getDevicePointer(),
                                                     &GaussianVolume->getDevicePointer(),
                                                     &AtomicGamma->getDevicePointer(),
                                                     &gtree->ovLevel->getDevicePointer(),
                                                     &gtree->ovVolume->getDevicePointer(),
                                                     &gtree->ovVsp->getDevicePointer(),
                                                     &gtree->ovVSfp->getDevicePointer(),
                                                     &gtree->ovGamma1i->getDevicePointer(),
                                                     &gtree->ovG->getDevicePointer(),
                                                     &gtree->ovDV1->getDevicePointer(),
                                                     &gtree->ovLastAtom->getDevicePointer(),
                                                     &gtree->ovRootIndex->getDevicePointer(),
                                                     &gtree->ovChildrenStartIndex->getDevicePointer(),
                                                     &gtree->ovChildrenCount->getDevicePointer()
        };
        cu.executeKernel(InitOverlapTreeKernel_1body_1, InitOverlapTreeKernel_1body_1Args, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute InitOverlapTreeCountKernel
    {
        // compute numbers of 2-body overlaps, that is children counts of 1-body overlaps
        unsigned int interactingTileSize = nb.getInteractingTiles().getSize();
        void *InitOverlapTreeCountKernelCutoffArgs[] = {&gtree->ovAtomTreePointer->getDevicePointer(),
                                                        &cu.getPosq().getDevicePointer(),
                                                        &GaussianExponent->getDevicePointer(),
                                                        &GaussianVolume->getDevicePointer(),
                                                        &nb.getInteractingTiles().getDevicePointer(),
                                                        &nb.getInteractionCount().getDevicePointer(),
                                                        &nb.getInteractingAtoms().getDevicePointer(),
                                                        &interactingTileSize,
                                                        &nb.getExclusionTiles().getDevicePointer(),
                                                        &gtree->ovChildrenCount->getDevicePointer()};

        unsigned int numAtomBlocks = (cu.getNumAtomBlocks() * (cu.getNumAtomBlocks() + 1) / 2);
        void *InitOverlapTreeCountKernelArgs[] = {&gtree->ovAtomTreePointer->getDevicePointer(),
                                                  &cu.getPosq().getDevicePointer(),
                                                  &GaussianExponent->getDevicePointer(),
                                                  &GaussianVolume->getDevicePointer(),
                                                  &numAtomBlocks,
                                                  &gtree->ovChildrenCount->getDevicePointer()};
        if(useCutoff){
            cu.executeKernel(InitOverlapTreeCountKernel, InitOverlapTreeCountKernelCutoffArgs, ov_work_group_size * num_compute_units, ov_work_group_size);
        }
        else{
            cu.executeKernel(InitOverlapTreeCountKernel, InitOverlapTreeCountKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);
        }
    }

    //Execute reduceovCountBufferKernel
    {
        // do a prefix sum of 2-body counts to compute children start indexes to store 2-body overlaps computed by InitOverlapTreeKernel below
        void *reduceovCountBufferKernelArgs[] = {&num_sections,
                                                 &gtree->ovTreePointer->getDevicePointer(),
                                                 &gtree->ovAtomTreePointer->getDevicePointer(),
                                                 &gtree->ovAtomTreeSize->getDevicePointer(),
                                                 &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                 &gtree->ovChildrenStartIndex->getDevicePointer(),
                                                 &gtree->ovChildrenCount->getDevicePointer(),
                                                 &gtree->ovChildrenCountTop->getDevicePointer(),
                                                 &gtree->ovChildrenCountBottom->getDevicePointer(),
                                                 &PanicButton->getDevicePointer()};
        cu.executeKernel(reduceovCountBufferKernel, reduceovCountBufferKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute InitOverlapTreeKernel
    {
        unsigned int interactingTileSize = nb.getInteractingTiles().getSize();
        unsigned int numAtomBlocks = (cu.getNumAtomBlocks() * (cu.getNumAtomBlocks() + 1) / 2);
        void *InitOverlapTreeKernelCutoffArgs[] = {&gtree->ovAtomTreePointer->getDevicePointer(),
                                                   &gtree->ovAtomTreeSize->getDevicePointer(),
                                                   &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                   &cu.getPosq().getDevicePointer(),
                                                   &GaussianExponent->getDevicePointer(),
                                                   &GaussianVolume->getDevicePointer(),
                                                   &AtomicGamma->getDevicePointer(),
                                                   &nb.getInteractingTiles().getDevicePointer(),
                                                   &nb.getInteractionCount().getDevicePointer(),
                                                   &nb.getInteractingAtoms().getDevicePointer(),
                                                   &interactingTileSize,
                                                   &nb.getExclusionTiles().getDevicePointer(),
                                                   &gtree->ovLevel->getDevicePointer(),
                                                   &gtree->ovVolume->getDevicePointer(),
                                                   &gtree->ovVsp->getDevicePointer(),
                                                   &gtree->ovVSfp->getDevicePointer(),
                                                   &gtree->ovGamma1i->getDevicePointer(),
                                                   &gtree->ovG->getDevicePointer(),
                                                   &gtree->ovDV1->getDevicePointer(),
                                                   &gtree->ovLastAtom->getDevicePointer(),
                                                   &gtree->ovRootIndex->getDevicePointer(),
                                                   &gtree->ovChildrenStartIndex->getDevicePointer(),
                                                   &gtree->ovChildrenCount->getDevicePointer(),
                                                   &gtree->ovChildrenCountTop->getDevicePointer(),
                                                   &gtree->ovChildrenCountBottom->getDevicePointer(),
                                                   &PanicButton->getDevicePointer()};

        void *InitOverlapTreeKernelArgs[] = {&gtree->ovAtomTreePointer->getDevicePointer(),
                                             &gtree->ovAtomTreeSize->getDevicePointer(),
                                             &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                             &cu.getPosq().getDevicePointer(),
                                             &GaussianExponent->getDevicePointer(),
                                             &GaussianVolume->getDevicePointer(),
                                             &AtomicGamma->getDevicePointer(),
                                             &numAtomBlocks,
                                             &gtree->ovLevel->getDevicePointer(),
                                             &gtree->ovVolume->getDevicePointer(),
                                             &gtree->ovVsp->getDevicePointer(),
                                             &gtree->ovVSfp->getDevicePointer(),
                                             &gtree->ovGamma1i->getDevicePointer(),
                                             &gtree->ovG->getDevicePointer(),
                                             &gtree->ovDV1->getDevicePointer(),
                                             &gtree->ovLastAtom->getDevicePointer(),
                                             &gtree->ovRootIndex->getDevicePointer(),
                                             &gtree->ovChildrenStartIndex->getDevicePointer(),
                                             &gtree->ovChildrenCount->getDevicePointer(),
                                             &gtree->ovChildrenCountTop->getDevicePointer(),
                                             &gtree->ovChildrenCountBottom->getDevicePointer(),
                                             &PanicButton->getDevicePointer()};
        if(useCutoff){
            cu.executeKernel(InitOverlapTreeKernel, InitOverlapTreeKernelCutoffArgs, ov_work_group_size * num_compute_units, ov_work_group_size);
        }
        else{
            cu.executeKernel(InitOverlapTreeKernel, InitOverlapTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);
        }
    }

    //Execute resetComputeOverlapTreeKernel
    {
        void *resetComputeOverlapTreeKernelArgs[] = {&num_sections,
                                                     &gtree->ovTreePointer->getDevicePointer(),
                                                     &gtree->ovProcessedFlag->getDevicePointer(),
                                                     &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                                     &gtree->ovAtomTreeSize->getDevicePointer(),
                                                     &gtree->ovLevel->getDevicePointer()};
        cu.executeKernel(resetComputeOverlapTreeKernel, resetComputeOverlapTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute ComputeOverlapTree_1passKernel
    {
        int temp_buffer_size = gtree->temp_buffer_size;
        void *ComputeOverlapTree_1passKernelArgs[] = {&num_sections,
                                                      &gtree->ovTreePointer->getDevicePointer(),
                                                      &gtree->ovAtomTreePointer->getDevicePointer(),
                                                      &gtree->ovAtomTreeSize->getDevicePointer(),
                                                      &gtree->NIterations->getDevicePointer(),
                                                      &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                      &gtree->ovAtomTreeLock->getDevicePointer(),
                                                      &cu.getPosq().getDevicePointer(),
                                                      &GaussianExponent->getDevicePointer(),
                                                      &GaussianVolume->getDevicePointer(),
                                                      &AtomicGamma->getDevicePointer(),
                                                      &gtree->ovLevel->getDevicePointer(),
                                                      &gtree->ovVolume->getDevicePointer(),
                                                      &gtree->ovVsp->getDevicePointer(),
                                                      &gtree->ovVSfp->getDevicePointer(),
                                                      &gtree->ovGamma1i->getDevicePointer(),
                                                      &gtree->ovG->getDevicePointer(),
                                                      &gtree->ovDV1->getDevicePointer(),
                                                      &gtree->ovLastAtom->getDevicePointer(),
                                                      &gtree->ovRootIndex->getDevicePointer(),
                                                      &gtree->ovChildrenStartIndex->getDevicePointer(),
                                                      &gtree->ovChildrenCount->getDevicePointer(),
                                                      &gtree->ovProcessedFlag->getDevicePointer(),
                                                      &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                                      &gtree->ovChildrenReported->getDevicePointer(),
                                                      &gtree->ovChildrenCountTop->getDevicePointer(),
                                                      &gtree->ovChildrenCountBottom->getDevicePointer(),
                                                      &temp_buffer_size,
                                                      &gtree->gvol_buffer_temp->getDevicePointer(),
                                                      &gtree->tree_pos_buffer_temp->getDevicePointer(),
                                                      &gtree->i_buffer_temp->getDevicePointer(),
                                                      &gtree->atomj_buffer_temp->getDevicePointer(),
                                                      &PanicButton->getDevicePointer()};
        cu.executeKernel(ComputeOverlapTree_1passKernel, ComputeOverlapTree_1passKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    PanicButton->download(pinnedPanicButtonMemory, false);
    cuEventRecord(downloadPanicButtonEvent, cu.getCurrentStream());

    //------------------------------------------------------------------------------------------------------------


    //------------------------------------------------------------------------------------------------------------
    // Volume energy function 1 (large radii)
    //

    //Execute resetSelfVolumesKernel
    {
        void *resetSelfVolumesArgs[] = {&num_sections,
                                        &gtree->ovTreePointer->getDevicePointer(),
                                        &gtree->ovAtomTreePointer->getDevicePointer(),
                                        &gtree->ovAtomTreeSize->getDevicePointer(),
                                        &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                        &gtree->ovChildrenStartIndex->getDevicePointer(),
                                        &gtree->ovChildrenCount->getDevicePointer(),
                                        &gtree->ovProcessedFlag->getDevicePointer(),
                                        &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                        &gtree->ovChildrenReported->getDevicePointer(),
                                        &PanicButton->getDevicePointer()};
        cu.executeKernel(resetSelfVolumesKernel, resetSelfVolumesArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //TODO: Panic Button?
    //check the result of the non-blocking read of PanicButton above
    cuEventSynchronize(downloadPanicButtonEvent);
    if (pinnedPanicButtonMemory[0] > 0) {
        hasInitializedKernels = false; //forces reinitialization
        cu.setForcesValid(false); //invalidate forces

        if (pinnedPanicButtonMemory[1] > 0) {
            gtree->hasExceededTempBuffer = true;//forces resizing of temp buffers
        }

        return 0.0;
    }

    //Execute computeSelfVolumesKernel
    {
        void *computeSelfVolumesKernelArgs[] = {&num_sections,
                                                &gtree->ovTreePointer->getDevicePointer(),
                                                &gtree->ovAtomTreePointer->getDevicePointer(),
                                                &gtree->ovAtomTreeSize->getDevicePointer(),
                                                &gtree->NIterations->getDevicePointer(),
                                                &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                &GaussianExponent->getDevicePointer(),
                                                &paddedNumAtoms,
                                                &gtree->ovLevel->getDevicePointer(),
                                                &gtree->ovVolume->getDevicePointer(),
                                                &gtree->ovVsp->getDevicePointer(),
                                                &gtree->ovVSfp->getDevicePointer(),
                                                &gtree->ovGamma1i->getDevicePointer(),
                                                &gtree->ovG->getDevicePointer(),
                                                &gtree->ovSelfVolume->getDevicePointer(),
                                                &gtree->ovVolEnergy->getDevicePointer(),
                                                &gtree->ovDV1->getDevicePointer(),
                                                &gtree->ovDV2->getDevicePointer(),
                                                &gtree->ovPF->getDevicePointer(),
                                                &gtree->ovLastAtom->getDevicePointer(),
                                                &gtree->ovRootIndex->getDevicePointer(),
                                                &gtree->ovChildrenStartIndex->getDevicePointer(),
                                                &gtree->ovChildrenCount->getDevicePointer(),
                                                &gtree->ovProcessedFlag->getDevicePointer(),
                                                &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                                &gtree->ovChildrenReported->getDevicePointer(),
                                                &gtree->ovAtomBuffer->getDevicePointer(),
                                                &gtree->gradBuffers_long->getDevicePointer(),
                                                &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                                &gtree->selfVolumeBuffer->getDevicePointer()};
        cu.executeKernel(computeSelfVolumesKernel, computeSelfVolumesKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute reduceSelfVolumesKernel_buffer
    {
        void *reduceSelfVolumesKernel_bufferArgs[] = {&numAtoms,
                                                      &paddedNumAtoms,
                                                      &num_sections,
                                                      &gtree->ovAtomTreePointer->getDevicePointer(),
                                                      &gtree->ovAtomBuffer->getDevicePointer(),
                                                      &gtree->gradBuffers_long->getDevicePointer(),
                                                      &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                                      &gtree->selfVolumeBuffer->getDevicePointer(),
                                                      &selfVolume->getDevicePointer(),
                                                      &GaussianVolume->getDevicePointer(),
                                                      &AtomicGamma->getDevicePointer(),
                                                      &grad->getDevicePointer()};
        cu.executeKernel(reduceSelfVolumesKernel_buffer, reduceSelfVolumesKernel_bufferArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute updateSelfVolumesForces
    {
        int update_energy = 1;
        void *updateSelfVolumesForcesKernelArgs[] ={&update_energy,
                                                    &numAtoms,
                                                    &paddedNumAtoms,
                                                    &gtree->ovAtomTreePointer->getDevicePointer(),
                                                    &gtree->ovVolEnergy->getDevicePointer(),
                                                    &grad->getDevicePointer(),
                                                    &cu.getForce().getDevicePointer(),
                                                    &cu.getEnergyBuffer().getDevicePointer()};
        cu.executeKernel(updateSelfVolumesForcesKernel, updateSelfVolumesForcesKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    vector<int> atom_pointer;
    vector<float> vol_energies;
    gtree->ovAtomTreePointer->download(atom_pointer);
    gtree->ovVolEnergy->download(vol_energies);
    double energy = 0;
    for (int i = 0; i < numParticles; i++) {
        int slot = atom_pointer[i];
        energy += vol_energies[slot];
    }

    //------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------
    // Self volumes, volume scaling parameters,
    // volume energy function 2 (small radii), surface area cavity energy function
    //

    //seeds tree with "negative" gammas and reduced radii
    //Execute InitOverlapTreeKernel_1body_2
    {
        int reset_tree_size = 0;
        void *InitOverlapTreeKernel_1body_2Args[] = {&paddedNumAtoms,
                                                     &num_sections,
                                                     &reset_tree_size,
                                                     &gtree->ovTreePointer->getDevicePointer(),
                                                     &gtree->ovNumAtomsInTree->getDevicePointer(),
                                                     &gtree->ovFirstAtom->getDevicePointer(),
                                                     &gtree->ovAtomTreeSize->getDevicePointer(),
                                                     &gtree->NIterations->getDevicePointer(),
                                                     &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                     &gtree->ovAtomTreePointer->getDevicePointer(),
                                                     &cu.getPosq().getDevicePointer(),
                                                     &radiusParam2->getDevicePointer(),
                                                     &gammaParam2->getDevicePointer(),
                                                     &ishydrogenParam->getDevicePointer(),
                                                     &GaussianExponent->getDevicePointer(),
                                                     &GaussianVolume->getDevicePointer(),
                                                     &AtomicGamma->getDevicePointer(),
                                                     &gtree->ovLevel->getDevicePointer(),
                                                     &gtree->ovVolume->getDevicePointer(),
                                                     &gtree->ovVsp->getDevicePointer(),
                                                     &gtree->ovVSfp->getDevicePointer(),
                                                     &gtree->ovGamma1i->getDevicePointer(),
                                                     &gtree->ovG->getDevicePointer(),
                                                     &gtree->ovDV1->getDevicePointer(),
                                                     &gtree->ovLastAtom->getDevicePointer(),
                                                     &gtree->ovRootIndex->getDevicePointer(),
                                                     &gtree->ovChildrenStartIndex->getDevicePointer(),
                                                     &gtree->ovChildrenCount->getDevicePointer()};
        cu.executeKernel(InitOverlapTreeKernel_1body_2, InitOverlapTreeKernel_1body_2Args, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute ResetRescanOverlapTreeKernel
    {
        void *ResetRescanOverlapTreeKernelArgs[] = {&num_sections,
                                                    &gtree->ovTreePointer->getDevicePointer(),
                                                    &gtree->ovAtomTreePointer->getDevicePointer(),
                                                    &gtree->ovAtomTreeSize->getDevicePointer(),
                                                    &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                    &gtree->ovProcessedFlag->getDevicePointer(),
                                                    &gtree->ovOKtoProcessFlag->getDevicePointer()};
        cu.executeKernel(ResetRescanOverlapTreeKernel, ResetRescanOverlapTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute InitRescanOverlapTreeKernel
    {
        void *InitRescanOverlapTreeKernelArgs[] = {&num_sections,
                                                   &gtree->ovTreePointer->getDevicePointer(),
                                                   &gtree->ovAtomTreeSize->getDevicePointer(),
                                                   &gtree->ovProcessedFlag->getDevicePointer(),
                                                   &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                                   &gtree->ovLevel->getDevicePointer()};
        cu.executeKernel(InitRescanOverlapTreeKernel, InitRescanOverlapTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute RescanOverlapTreeKernel
    {
        void *RescanOverlapTreeKernelArgs[] = {&num_sections,
                                               &gtree->ovTreePointer->getDevicePointer(),
                                               &gtree->ovAtomTreePointer->getDevicePointer(),
                                               &gtree->ovAtomTreeSize->getDevicePointer(),
                                               &gtree->NIterations->getDevicePointer(),
                                               &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                               &gtree->ovAtomTreeLock->getDevicePointer(),
                                               &cu.getPosq().getDevicePointer(),
                                               &GaussianExponent->getDevicePointer(),
                                               &GaussianVolume->getDevicePointer(),
                                               &AtomicGamma->getDevicePointer(),
                                               &gtree->ovLevel->getDevicePointer(),
                                               &gtree->ovVolume->getDevicePointer(),
                                               &gtree->ovVsp->getDevicePointer(),
                                               &gtree->ovVSfp->getDevicePointer(),
                                               &gtree->ovGamma1i->getDevicePointer(),
                                               &gtree->ovG->getDevicePointer(),
                                               &gtree->ovDV1->getDevicePointer(),
                                               &gtree->ovLastAtom->getDevicePointer(),
                                               &gtree->ovRootIndex->getDevicePointer(),
                                               &gtree->ovChildrenStartIndex->getDevicePointer(),
                                               &gtree->ovChildrenCount->getDevicePointer(),
                                               &gtree->ovProcessedFlag->getDevicePointer(),
                                               &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                               &gtree->ovChildrenReported->getDevicePointer()};
        cu.executeKernel(RescanOverlapTreeKernel, RescanOverlapTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute resetSelfVolumesKernel
    {
        void *resetSelfVolumesArgs[] = {&num_sections,
                                        &gtree->ovTreePointer->getDevicePointer(),
                                        &gtree->ovAtomTreePointer->getDevicePointer(),
                                        &gtree->ovAtomTreeSize->getDevicePointer(),
                                        &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                        &gtree->ovChildrenStartIndex->getDevicePointer(),
                                        &gtree->ovChildrenCount->getDevicePointer(),
                                        &gtree->ovProcessedFlag->getDevicePointer(),
                                        &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                        &gtree->ovChildrenReported->getDevicePointer(),
                                        &PanicButton->getDevicePointer()};
        cu.executeKernel(resetSelfVolumesKernel, resetSelfVolumesArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    // zero self-volume accumulator
    //Executing resetBufferKernel
    {
        void *resetBufferKernelArgs[] = {&paddedNumAtoms,
                                         &num_sections,
                                         &gtree->ovAtomBuffer->getDevicePointer(),
                                         &gtree->selfVolumeBuffer->getDevicePointer(),
                                         &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                         &gtree->gradBuffers_long->getDevicePointer()};
        cu.executeKernel(resetBufferKernel, resetBufferKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute computeSelfVolumesKernel
    {
        void *computeSelfVolumesKernelArgs[] = {&num_sections,
                                                &gtree->ovTreePointer->getDevicePointer(),
                                                &gtree->ovAtomTreePointer->getDevicePointer(),
                                                &gtree->ovAtomTreeSize->getDevicePointer(),
                                                &gtree->NIterations->getDevicePointer(),
                                                &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                &GaussianExponent->getDevicePointer(),
                                                &paddedNumAtoms,
                                                &gtree->ovLevel->getDevicePointer(),
                                                &gtree->ovVolume->getDevicePointer(),
                                                &gtree->ovVsp->getDevicePointer(),
                                                &gtree->ovVSfp->getDevicePointer(),
                                                &gtree->ovGamma1i->getDevicePointer(),
                                                &gtree->ovG->getDevicePointer(),
                                                &gtree->ovSelfVolume->getDevicePointer(),
                                                &gtree->ovVolEnergy->getDevicePointer(),
                                                &gtree->ovDV1->getDevicePointer(),
                                                &gtree->ovDV2->getDevicePointer(),
                                                &gtree->ovPF->getDevicePointer(),
                                                &gtree->ovLastAtom->getDevicePointer(),
                                                &gtree->ovRootIndex->getDevicePointer(),
                                                &gtree->ovChildrenStartIndex->getDevicePointer(),
                                                &gtree->ovChildrenCount->getDevicePointer(),
                                                &gtree->ovProcessedFlag->getDevicePointer(),
                                                &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                                &gtree->ovChildrenReported->getDevicePointer(),
                                                &gtree->ovAtomBuffer->getDevicePointer(),
                                                &gtree->gradBuffers_long->getDevicePointer(),
                                                &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                                &gtree->selfVolumeBuffer->getDevicePointer()};
        cu.executeKernel(computeSelfVolumesKernel, computeSelfVolumesKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //update energyBuffer with volume energy 2
    //Execute reduceSelfVolumesKernel_buffer
    {
        void *reduceSelfVolumesKernel_bufferArgs[] = {&numAtoms,
                                                      &paddedNumAtoms,
                                                      &num_sections,
                                                      &gtree->ovAtomTreePointer->getDevicePointer(),
                                                      &gtree->ovAtomBuffer->getDevicePointer(),
                                                      &gtree->gradBuffers_long->getDevicePointer(),
                                                      &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                                      &gtree->selfVolumeBuffer->getDevicePointer(),
                                                      &selfVolume->getDevicePointer(),
                                                      &GaussianVolume->getDevicePointer(),
                                                      &AtomicGamma->getDevicePointer(),
                                                      &grad->getDevicePointer()};
        cu.executeKernel(reduceSelfVolumesKernel_buffer, reduceSelfVolumesKernel_bufferArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute updateSelfVolumesForces
    {
        int update_energy = 1;
        void *updateSelfVolumesForcesKernelArgs[] ={&update_energy,
                                                    &numAtoms,
                                                    &paddedNumAtoms,
                                                    &gtree->ovAtomTreePointer->getDevicePointer(),
                                                    &gtree->ovVolEnergy->getDevicePointer(),
                                                    &grad->getDevicePointer(),
                                                    &cu.getForce().getDevicePointer(),
                                                    &cu.getEnergyBuffer().getDevicePointer()};
        cu.executeKernel(updateSelfVolumesForcesKernel, updateSelfVolumesForcesKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    gtree->ovAtomTreePointer->download(atom_pointer);
    gtree->ovVolEnergy->download(vol_energies);
    energy = 0;
    for (int i = 0; i < numParticles; i++) {
        int slot = atom_pointer[i];
        energy += vol_energies[slot];
    }

    return 0.0;
}

void CudaCalcGKCavitationForceKernel::copyParametersToContext(ContextImpl &context, const AmoebaGKCavitationForce &force) {
    if (force.getNumParticles() != numParticles) {
        cout << force.getNumParticles() << " != " << numParticles << endl; //Debug
        throw OpenMMException("copyParametersToContext: GKCavitation plugin does not support changing the number of atoms.");
    }
    if (numParticles == 0)
        return;
    for (int i = 0; i < numParticles; i++) {
        double radius, gamma;
        bool ishydrogen;
        force.getParticleParameters(i, radius, gamma, ishydrogen);
        if (pow(radiusVector2[i] - radius, 2) > 1.e-6) {
            throw OpenMMException("updateParametersInContext: GKCavitation plugin does not support changing atomic radii.");
        }
        int h = ishydrogen ? 1 : 0;
        if (ishydrogenVector[i] != h) {
            throw OpenMMException(
                    "updateParametersInContext: GKCavitation plugin does not support changing heavy/hydrogen atoms.");
        }
        double g = ishydrogen ? 0 : gamma / roffset;
        gammaVector1[i] = (float) g;
        gammaVector2[i] = (float) -g;
    }
    gammaParam1->upload(gammaVector1);
    gammaParam2->upload(gammaVector2);
}

