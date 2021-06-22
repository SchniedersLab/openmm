/* Portions copyright (c) 2006 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <vector>
#include "AmoebaReferenceGKCavitationForce.h"

using namespace OpenMM;

double AmoebaReferenceGKCavitationForce::calculateForceAndEnergy(vector<RealVec> &pos, vector<RealVec> &force,
                                                                 int numParticles,
                                                                 vector<int> ishydrogen, vector<RealOpenMM> radii_large,
                                                                 vector<RealOpenMM> radii_vdw,
                                                                 vector<RealOpenMM> gammas,
                                                                 double roffset, vector<RealVec> vol_force,
                                                                 vector<RealOpenMM> vol_dv,
                                                                 vector<RealOpenMM> free_volume,
                                                                 vector<RealOpenMM> self_volume) {
    // Create and saves GaussVol instance
    // radii, volumes, etc. will be set in execute()
    gvol = new GaussVol(numParticles, ishydrogen);

    //sequence: volume1->volume2
    //weights
    double w_evol = 1.0;
    double energy;
    vector<RealOpenMM> nu(numParticles);

    // volume energy function 1 (large radii)
    RealOpenMM volume1, vol_energy1;
    gvol->setRadii(radii_large);

    vector<RealOpenMM> volumes_large(numParticles);
    for (int i = 0; i < numParticles; i++) {
        volumes_large[i] = ishydrogen[i] > 0.0f ? 0.0f : 4.0f * M_PI * pow(radii_large[i], 3) / 3.0f;
    }
    gvol->setVolumes(volumes_large);

    for (int i = 0; i < numParticles; i++) {
        nu[i] = gammas[i] / roffset;
    }
    gvol->setGammas(nu);
    gvol->compute_tree(pos);
    gvol->compute_volume(pos, volume1, vol_energy1, vol_force, vol_dv, free_volume, self_volume);

    //returns energy and gradients from volume energy function
    for (int i = 0; i < numParticles; i++) {
        if (!ishydrogen[i]) {
           force[i] += vol_force[i] * w_evol;
        }
    }

    energy = vol_energy1 * w_evol;

    // volume energy function 2 (small radii)
    double vol_energy2, volume2;
    gvol->setRadii(radii_vdw);

    vector<RealOpenMM> volumes_vdw(numParticles);
    for (int i = 0; i < numParticles; i++) {
        volumes_vdw[i] = ishydrogen[i] > 0.0f ? 0.0f : 4.0f * M_PI * pow(radii_vdw[i], 3) / 3.0f;
    }

    gvol->setVolumes(volumes_vdw);

    for (int i = 0; i < numParticles; i++) {
        nu[i] = -gammas[i] / roffset;
    }

    gvol->setGammas(nu);
    gvol->rescan_tree_volumes(pos);
    gvol->compute_volume(pos, volume2, vol_energy2, vol_force, vol_dv, free_volume, self_volume);

    for (int i = 0; i < numParticles; i++) {
        if (!ishydrogen[i]) {
           force[i] += vol_force[i] * w_evol;
        }
    }

    energy += vol_energy2 * w_evol;

    return energy;
}

