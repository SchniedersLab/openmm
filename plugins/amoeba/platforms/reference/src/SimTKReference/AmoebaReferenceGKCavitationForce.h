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
#ifndef OPENMM_AMOEBAREFERENCEGKCAVITATIONFORCE_H
#define OPENMM_AMOEBAREFERENCEGKCAVITATIONFORCE_H

#include <RealVec.h>
#include "gaussvol.h"

using std::vector;

namespace OpenMM {
    class AmoebaReferenceGKCavitationForce {
    public:
        AmoebaReferenceGKCavitationForce() {
            gvol = 0;
        };

        ~AmoebaReferenceGKCavitationForce() {
            if (gvol) delete gvol;
        };

        double calculateForceAndEnergy(vector<RealVec> &pos, vector<RealVec> &force, int numParticles,
                                       vector<int> ishydrogen, vector<RealOpenMM> radii_large,
                                       vector<RealOpenMM> radii_vdw, vector<RealOpenMM> gammas,
                                       double roffset, vector<RealVec> vol_force, vector<RealOpenMM> vol_dv,
                                       vector<RealOpenMM> free_volume,
                                       vector<RealOpenMM> self_volume);

    private:
        // gaussvol instance
        GaussVol *gvol;
    };
}

#endif //OPENMM_AMOEBAREFERENCEGKCAVITATIONFORCE_H
