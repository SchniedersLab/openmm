/* -------------------------------------------------------------------------- *
 *                            OpenMM-GKCavitation                             *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/internal/AmoebaGKCavitationForceImpl.h"
#include "openmm/amoebaKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <map>
#include <set>
#include <sstream>

using namespace OpenMM;
using namespace std;

AmoebaGKCavitationForceImpl::AmoebaGKCavitationForceImpl(const AmoebaGKCavitationForce& owner) : owner(owner) {
}

AmoebaGKCavitationForceImpl::~AmoebaGKCavitationForceImpl() {
}

void AmoebaGKCavitationForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcGKCavitationForceKernel::Name(), context);
    kernel.getAs<CalcGKCavitationForceKernel>().initialize(context.getSystem(), owner);
}

double AmoebaGKCavitationForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
  if ((groups&(1<<owner.getForceGroup())) != 0)
    return kernel.getAs<CalcGKCavitationForceKernel>().execute(context, includeForces, includeEnergy);
  return 0.0;
}

std::vector<std::string> AmoebaGKCavitationForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcGKCavitationForceKernel::Name());
    return names;
}

void AmoebaGKCavitationForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcGKCavitationForceKernel>().copyParametersToContext(context, owner);
}
