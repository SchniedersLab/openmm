/* -------------------------------------------------------------------------- *
 *                              OpenMM-GKCavitation                                 *
 * -------------------------------------------------------------------------- */

/**
 * This tests the CUDA implementation of GKNPForce.
 */

#define _USE_MATH_DEFINES // Needed to get M_PI
#include "openmm/AmoebaGKCavitationForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/NonbondedForce.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerAmoebaCudaKernelFactories();

static struct MyAtomInfo {
    const char *pdb;
    double mass, vdwRadiusInAng, gamma;
    bool isHydrogen;
    double initPosInAng[3];
} atoms[] = {
        // Atom name, mass, vdwRad (A), gamma, (kcal/mol/A^2), isHydrogen, initPos
        {" C ", 12.00, 1.91, 0.103, false, -0.76556335, 0.00001165,  -0.00000335},
        {" C ", 12.00, 1.91, 0.103, false, 0.76556335,  -0.00001165, 0.00000335},
        {" H ", 1.00,  1.48, 0,     true,  -1.16801233, 0.65545698,  0.78074120},
        {" H ", 1.00,  1.48, 0,     true,  -1.16800941, 0.34844426,  -0.95800528},
        {" H ", 1.00,  1.48, 0,     true,  -1.16803492, -1.00384801, 0.17724878},
        {" H ", 1.00,  1.48, 0,     true,  1.16800941,  -0.34844426, 0.95800528},
        {" H ", 1.00,  1.48, 0,     true,  1.16803492,  1.00384801,  -0.17724878},
        {" H ", 1.00,  1.48, 0,     true,  1.16801233,  -0.65545698, -0.78074120},
        {""} // end of list
};

void testForce() {

    System system;
    NonbondedForce *nb = new NonbondedForce();
    AmoebaGKCavitationForce* force = new AmoebaGKCavitationForce();
    force->setNonbondedMethod(AmoebaGKCavitationForce::NoCutoff);//NoCutoff also accepted
    force->setCutoffDistance(1.0);
    system.addForce(nb);
    system.addForce(force);

    int numParticles = 8;
    vector<Vec3> positions;

    // Constants for unit/energy conversions
    double surfaceTension = 0.103;
    double ang2nm = 0.1;
    double kcalmol2kjmol = 4.184;
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(atoms[i].mass);
        positions.push_back(
                Vec3(atoms[i].initPosInAng[0], atoms[i].initPosInAng[1], atoms[i].initPosInAng[2]) * ang2nm);
        atoms[i].vdwRadiusInAng *= ang2nm;
        atoms[i].gamma *= kcalmol2kjmol / (ang2nm * ang2nm);
        nb->addParticle(0.0, 0.0, 0.0);
        force->addParticle(atoms[i].vdwRadiusInAng, atoms[i].gamma, atoms[i].isHydrogen);
        force->getParticleParameters(i, atoms[i].vdwRadiusInAng, atoms[i].gamma, atoms[i].isHydrogen);
    }

    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");

    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces | State::Positions);

    double energy1 = state.getPotentialEnergy();
    double surfaceArea = (energy1 / kcalmol2kjmol) / surfaceTension;
    double surfaceAreaEnergy = surfaceArea * surfaceTension;

    // TODO: Replace these with assert statements.
    //  Force Field X Values:
    //  Surface Area:          62.426 (Ang^2)
    //  Surface Area Energy:    6.430 (kcal/mol)

    cout << endl;
    cout << std::setw(25) << std::left << "Surface Area: " << std::fixed << surfaceArea << " (Ang^2)" << endl;
    cout << std::setw(25) << std::left << "Surface Area Energy:  " << surfaceAreaEnergy << " (kcal/mol)" << endl << endl;

//    cout << "Forces: " << endl;
//    for(int i = 0; i < numParticles; i++) {
//        cout << "FW: " << i << " " << state.getForces()[i][0] << " " << state.getForces()[i][1] << " "
//             << state.getForces()[i][2] << " " << endl;
//    }

    // Validate force by moving an atom
#ifdef NOTNOW
    double offset = 1.0e-3;
    int pmove = 0;
    int direction = 0;
    positions[pmove][direction] += offset;
    context.setPositions(positions);
    double energy2 = context.getState(State::Energy).getPotentialEnergy();
    double de = -state.getForces()[pmove][direction]*offset;
    std::cout << "Energy: " <<  energy2  << std::endl;
    std::cout << "Energy Change: " <<  energy2 - energy1  << std::endl;
    std::cout << "Energy Change from Gradient: " <<  de  << std::endl;
#endif

}

int main() {
    try {
        registerAmoebaCudaKernelFactories();
        testForce();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

