/*
  Copyright (C) 2010-2018 The ESPResSo project
  Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
    Max-Planck-Institute for Polymer Research, Theory Group

  This file is part of ESPResSo.

  ESPResSo is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ESPResSo is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
//

/** \file iccp3m.hpp

    ICCP3M is a method that allows to take into account the influence
    of arbitrarily shaped dielectric interfaces.  The dielectric
    properties of a dielectric medium in the bulk of the simulation
    box are taken into account by reproducing the jump in the electric
    field at the inface with charge surface segments. The charge
    density of the surface segments have to be determined
    self-consistently using an iterative scheme.  It can at presently
    - despite its name - be used with P3M, ELCP3M, MMM2D and MMM1D.
    For details see:<br> S. Tyagi, M. Suzen, M. Sega, C. Holm,
    M. Barbosa: A linear-scaling method for computing induced charges
    on arbitrary dielectric boundaries in large system simulations
    (Preprint)

    To set up ICCP3M first the dielectric boundary has to be modelled
    by espresso particles 0..n where n has to be passed as a parameter
    to ICCP3M. This is still a bit inconvenient, as it forces the user
    to reserve the first n particle ids to wall charges, but as the
    other parts of espresso do not suffer from a limitation like this,
    it can be tolerated.

    For the determination of the induced charges only the forces
    acting on the induced charges has to be determined. As P3M an the
    other coulomb solvers calculate all mutual forces, the force
    calculation was modified to avoid the calculation of the short
    range part of the source-source force calculation.  For different
    particle data organisation schemes this is performed differently.
    */

#ifndef CORE_ICCP3M_HPP
#define CORE_ICCP3M_HPP

#include "config.hpp"
#include "partCfg_global.hpp"

#if defined(ELECTROSTATICS)

#include "Vector.hpp"
#include "iccShape.hpp"
#include <queue>
#include <list>

/* iccp3m store data struct */
struct iccp3m_data_struct {
  int largestID = 0;
  NewParticle reducedPart;
  std::list<std::vector<int>> trackList;
  std::queue<std::vector<NewParticle>> newParticleData;
  std::vector<double> iccCharges;
  std::set<int> missingIDs;
};

/* iccp3m data structures*/
struct iccp3m_struct {
  bool active = false;
  int numMissingIDs = 0;
  int n_icc;                  /* Last induced id (can not be smaller than 2) */
  int num_iteration = 30;    /* Number of max iterations                    */
  double eout = 1;           /* Dielectric constant of the bulk             */
  double convergence = 1e-2; /* Convergence criterion                       */
  Vector3d ext_field = {0, 0, 0}; /* External field */
  double relax = 0.7; /* relaxation parameter for iterative */
  int citeration = 0; /* current number of iterations*/
  int set_flag =
      0; /* flag that indicates if ICCP3M has been initialized properly
          */
  int first_id = 0;
  double maxCharge = 0.;
  double minCharge = 0.;
  std::vector<iccShape *> iccTypes;

  template <typename Archive>
  void serialize(Archive &ar, long int /* version */) {
    ar &n_icc;
    ar &num_iteration;
    ar &first_id;
    ar &convergence;
    ar &eout;
    ar &relax;
    ar &ext_field;
    ar &citeration;
    ar &set_flag;
    ar &active;
    ar &minCharge;
    ar &maxCharge;
  }
};
extern iccp3m_struct iccp3m_cfg; /* global variable with ICCP3M configuration */
extern iccp3m_data_struct iccp3m_data;

/** The main iterative scheme, where the surface element charges are calculated
 * self-consistently.
 */
int iccp3m_iteration();

void c_splitParticles(PartCfg &partCfg, bool force);
void c_reduceParticle();

void c_checkSet(int ID);

int c_addTypeWall(Vector3d cutoff,
                  bool useTrans,
                  double transMatrix[9],
                  double invMatrix[9]);

int c_addTypeCylinder(Vector3d center,
                      Vector3d axis,
                      double length,
                      double radius,
                      double direction,
                      Vector3d cutoff,
                      bool useTrans,
                      double * transMatrix,
                      double * invMatrix);

int c_addTypeTorus(Vector3d center, Vector3d axis, double length, double radius, double smoothingRadius, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix);
int c_addTypeInterface(Vector3d center, double radius, double smoothingRadius, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix);

int c_outputVTK(char * filename, PartCfg & partCfg);
int c_outputParticle(char * filename, PartCfg & partCfg);

/** check sanity of parameters for use with ICCP3M
 */
int iccp3m_sanity_check();

#endif /* ELECTROSTATICS */

#endif /* ICCP3M_H */
