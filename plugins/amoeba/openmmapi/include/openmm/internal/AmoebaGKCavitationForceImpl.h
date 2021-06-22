#ifndef OPENMM_GKCAVITATIONFORCEIMPL_H_
#define OPENMM_GKCAVITATIONFORCEIMPL_H_

/* -------------------------------------------------------------------------- *
 *                             OpenMM-GKCavitation                            *
 * -------------------------------------------------------------------------- */

#include "openmm/AmoebaGKCavitationForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <string>

namespace OpenMM {

class System;

/**
 * This is the internal implementation of GKCavitationForce.
 */

class AmoebaGKCavitationForceImpl : public ForceImpl {
public:
    AmoebaGKCavitationForceImpl(const AmoebaGKCavitationForce& owner);
    ~AmoebaGKCavitationForceImpl();
    void initialize(ContextImpl& context);
    const AmoebaGKCavitationForce& getOwner() const {
        return owner;
    }
    void updateContextState(ContextImpl& context) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(ContextImpl& context,  bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(ContextImpl& context);
private:
    const AmoebaGKCavitationForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace OpenMM

#endif /*OPENMM_GKCAVITATIONFORCEIMPL_H_*/
