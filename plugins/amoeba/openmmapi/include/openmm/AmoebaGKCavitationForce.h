#ifndef OPENMM_GKCAVITATIONFORCE_H_
#define OPENMM_GKCAVITATIONFORCE_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM-GKCavitation                                        *
 * -------------------------------------------------------------------------- */

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <vector>
#include "internal/windowsExportAmoeba.h"

//use nm and kj
#define ANG (0.1f)
#define ANG2 (.01f)
#define ANG3 (0.001f)

//radius offset for surf energy calc.
#define GKCAV_RADIUS_INCREMENT (0.0005f)


namespace OpenMM {

/* This class implements the GKCavitation implicit solvation model */

class OPENMM_EXPORT_AMOEBA AmoebaGKCavitationForce : public OpenMM::Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,
        /**
         * Interactions beyond the cutoff distance are ignored.
         */
        CutoffNonPeriodic = 1,
        /**
         * Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of
         * each other particle.  Interactions beyond the cutoff distance are ignored.
         */
        CutoffPeriodic = 2,
    };
    /**
     * Create an GKCavitationForce.
     */
    AmoebaGKCavitationForce();
    /**
     * Add a particle to GKCavitation
     *
     * This should be called once for each particle in the System. When it is called for the i'th time, 
     * it specifies the parameters for the i'th particle.
     *
     * @param radius      the van der Waals radius of the particle, measured in nm
     * @param gamma       the surface tension parameter, measured in kJ/mol/nm^2
     * @param vdw_alpha   van der Waals solute-solvent interaction energy parameter
     * @param charge      electrostatic charge
     * @param ishydrogen  if true, this particle is a hydrogen atom (does not contribute to volume)
     * @return the index of the particle that was added
     */
    int addParticle(double radius, double gamma, bool ishydrogen);

    /**
     * Modify the parameters of a particle to GKCavitation
     *
     * @param index       the index of the particle
     * @param radius      the van der Waals radius of the particle, measured in nm
     * @param gamma       the surface tension parameter, measured in kJ/mol/nm^2
     * @param vdw_alpha   van der Waals solute-solvent interaction energy parameter
     * @param charge      electrostatic charge
     * @param ishydrogen  if true, this particle is a hydrogen atom (does not contribute to volume)
     * @return the index of the particle that was added
     */
    void setParticleParameters(int index, double radius, double gamma, bool ishydrogen);
    
    /**
     * Get the GKCavitation parameters for a particle
     * 
     * @param index       the index of the particle
     * @param radius      the van der Waals radius of the particle, measured in nm
     * @param gamma       the surface tension parameter, measured in kJ/mol/nm^2
     * @param vdw_alpha   van der Waals solute-solvent interaction energy parameter
     * @param charge      electric charge
     * @param ishydrogen  if true, this particle is a hydrogen atom
     */
    void getParticleParameters(int index, double& radius, double& gamma, bool& ishydrogen) const;
    /**
     * Get the number of particles defined for GKCavitation
     */
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Get the method used for handling long range nonbonded interactions.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long range nonbonded interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);
    
    void updateParametersInContext(Context& context);


protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    std::vector<ParticleInfo> particles;
    NonbondedMethod nonbondedMethod;
    double cutoffDistance;
};

/**
 * This is an internal class used to record information about particles.
 * @private
 */
class AmoebaGKCavitationForce::ParticleInfo {
 public:
  bool ishydrogen;
  double radius, gamma;
  ParticleInfo() {
    ishydrogen = false;
    radius = 0.15;
    gamma = 0.0;
  }
 ParticleInfo(double radius, double gamma, bool ishydrogen) :
  radius(radius), gamma(gamma), ishydrogen(ishydrogen) {  }
 };
 
} // namespace OpenMM

#endif /*OPENMM_GKCAVITATIONFORCE_H_*/
