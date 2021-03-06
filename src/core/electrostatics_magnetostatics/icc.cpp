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

/** \file iccp3m.cpp
  Detailed Information about the method is included in the corresponding header
  file \ref iccp3m.hpp.
*/

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <set>
#include <fstream>

#include "electrostatics_magnetostatics/elc.hpp"
#include "electrostatics_magnetostatics/mmm1d.hpp"
#include "electrostatics_magnetostatics/mmm2d.hpp"
#include "electrostatics_magnetostatics/p3m.hpp"
#include "electrostatics_magnetostatics/p3m_gpu.hpp"
#include "icc.hpp"

#include "communication.hpp"

#include "cells.hpp"
#include "config.hpp"
#include "forces.hpp"
#include "global.hpp"
#include "initialize.hpp"
#include "nonbonded_interactions/nonbonded_interaction_data.hpp"
#include "particle_data.hpp"
#include "partCfg_global.hpp"
#include "grid.hpp"

#include "short_range_loop.hpp"
#include "utils/NoOp.hpp"

#include "iccShape.hpp"
#include "iccWall.hpp"
#include "iccCylinder.hpp"
#include "iccTorus.hpp"
#include "iccInterface.hpp"

#ifdef ELECTROSTATICS

iccp3m_struct iccp3m_cfg;
iccp3m_data_struct iccp3m_data;

/* functions that are used in icc* to compute the electric field acting on the
 * induced charges, excluding forces other than the electrostatic ones */
void init_forces_iccp3m();
void calc_long_range_forces_iccp3m();

inline void init_local_particle_force_iccp3m(Particle *part);
inline void init_ghost_force_iccp3m(Particle *part);

/** Calculation of the electrostatic forces between source charges (= real
 * charges) and wall charges. For each electrostatic method the proper functions
 * for short and long range parts are called. Long Range Parts are calculated
 * directly, short range parts need helper functions according to the particle
 * data organisation. A modified version of \ref force_calc in \ref forces.hpp.
 */
void force_calc_iccp3m();

/** Variant of add_non_bonded_pair_force where only coulomb
 *  contributions are calculated   */
inline void add_non_bonded_pair_force_iccp3m(Particle *p1, Particle *p2,
                                             double d[3], double dist,
                                             double dist2) {
  /* IA_parameters *ia_params = get_ia_param(p1->p.type,p2->p.type);*/
  double force[3] = {0, 0, 0};

  FORCE_TRACE(fprintf(stderr, "%d: interaction %d<->%d dist %f\n", this_node,
                      p1->p.identity, p2->p.identity, dist));

  /***********************************************/
  /* long range electrostatics                   */
  /***********************************************/

  /* real space coulomb */
  auto const q1q2 = p1->p.q * p2->p.q;
  switch (coulomb.method) {
#ifdef P3M
  case COULOMB_ELC_P3M:
    if (q1q2)
      p3m_add_pair_force(q1q2, d, dist2, dist, force);
    break;
  case COULOMB_P3M_GPU:
  case COULOMB_P3M:
    if (q1q2)
      p3m_add_pair_force(q1q2, d, dist2, dist, force);
    break;
#endif /* P3M */
  case COULOMB_MMM1D:
    if (q1q2)
      add_mmm1d_coulomb_pair_force(q1q2, d, dist2, dist, force);
    break;
  case COULOMB_MMM2D:
    if (q1q2)
      add_mmm2d_coulomb_pair_force(q1q2, d, dist2, dist, force);
    break;
  default:
    break;
  }

  /***********************************************/
  /* add total nonbonded forces to particle      */
  /***********************************************/
  for (int j = 0; j < 3; j++) {
    p1->f.f[j] += force[j];
    p2->f.f[j] -= force[j];
  }
  /***********************************************/
}

int iccp3m_iteration() {
  if (!iccp3m_cfg.active)
    return 0;

  iccp3m_sanity_check();

  if ((iccp3m_cfg.eout <= 0)) {
    runtimeErrorMsg()
        << "ICCP3M: nonpositive dielectric constant is not allowed. Put a "
           "decent exception here\n";
  }

  auto const pref = 1.0 / (coulomb.prefactor * 6.283185307);
  iccp3m_cfg.citeration = 0;

  double globalmax = 1e100;

  for (int j = 0; j < iccp3m_cfg.num_iteration; j++) {
    double hmax = 0.;

    force_calc_iccp3m(); /* Calculate electrostatic forces (SR+LR) excluding
                            source source interaction*/
    ghost_communicator(&cell_structure.collect_ghost_force_comm);

    double diff = 0;

    for (auto &p : local_cells.particles()) {
      if (p.p.identity < iccp3m_cfg.n_icc + iccp3m_cfg.first_id + iccp3m_cfg.numMissingIDs &&
          p.p.identity >= iccp3m_cfg.first_id) {
        auto const id = p.p.identity - iccp3m_cfg.first_id;
        /* the dielectric-related prefactor: */
        auto const del_eps = (p.adapICC.eps - iccp3m_cfg.eout) /
                             (p.adapICC.eps + iccp3m_cfg.eout);
        /* calculate the electric field at the certain position */
        auto const E = p.f.f / p.p.q + iccp3m_cfg.ext_field;

        if (E[0] == 0 && E[1] == 0 && E[2] == 0) {
          runtimeErrorMsg()
              << "ICCP3M found zero electric field on a charge. This must "
                 "never happen";
        }

        /* recalculate the old charge density */
        auto const hold = p.p.q / p.adapICC.area;
        /* determine if it is higher than the previously highest charge
         * density */
        hmax = std::max(hmax, std::abs(hold));

        auto const f1 = del_eps * pref * (E * p.adapICC.normal);
        auto const f2 = (p.adapICC.sigma > 0.)
                            ? (2 * iccp3m_cfg.eout) /
                                  (iccp3m_cfg.eout + p.adapICC.eps) *
                                  (p.adapICC.sigma)
                            : 0.;
        /* relative variation: never use an estimator which can be negative
         * here */
        auto const hnew =
            (1. - iccp3m_cfg.relax) * hold + (iccp3m_cfg.relax) * (f1 + f2);

        /* Take the largest error to check for convergence */
        auto const relative_difference =
            std::abs(1 * (hnew - hold) / (hmax + std::abs(hnew + hold)));

        diff = std::max(diff, relative_difference);

        p.p.q = hnew * p.adapICC.area;

        /* check if the charge now is more than 1e6, to determine if ICC still
         * leads to reasonable results */
        /* this is kind a arbitrary measure but, does a good job spotting
         * divergence !*/
        if (std::abs(p.p.q) > 1e6) {
          runtimeErrorMsg()
              << "too big charge assignment in iccp3m! q >1e6 , assigned "
                 "charge= " << p.p.q << "\n";

          diff = 1e90; /* A very high value is used as error code */
          break;
        }
      }
    } /* cell particles */
    /* Update charges on ghosts. */
    ghost_communicator(&cell_structure.exchange_ghosts_comm);

    iccp3m_cfg.citeration++;

    MPI_Allreduce(&diff, &globalmax, 1, MPI_DOUBLE, MPI_MAX, comm_cart);

    // std::cerr << "iccIter: " << j << " - " << diff << '\n';

    if (globalmax < iccp3m_cfg.convergence)
      break;
    if (diff > 1e89) {
      return iccp3m_cfg.citeration++;
    }
  } /* iteration */

  if (globalmax > iccp3m_cfg.convergence) {
    runtimeErrorMsg()
        << "ICC failed to converge in the given number of maximal steps.";
  }

  on_particle_charge_change();

  return iccp3m_cfg.citeration;
}

void force_calc_iccp3m() {
  init_forces_iccp3m();

  short_range_loop(Utils::NoOp{}, [](Particle &p1, Particle &p2, Distance &d) {
    /* calc non bonded interactions */
    add_non_bonded_pair_force_iccp3m(&(p1), &(p2), d.vec21.data(),
                                     sqrt(d.dist2), d.dist2);
  });

  calc_long_range_forces_iccp3m();
}

void init_forces_iccp3m() {
  for (auto &p : local_cells.particles()) {
    p.f = ParticleForce{};
  }

  for (auto &p : ghost_cells.particles()) {
    p.f = ParticleForce{};
  }
}

void calc_long_range_forces_iccp3m() {
#ifdef ELECTROSTATICS
  /* calculate k-space part of electrostatic interaction. */
  if (!(coulomb.method == COULOMB_ELC_P3M ||
        coulomb.method == COULOMB_P3M_GPU || coulomb.method == COULOMB_P3M ||
        coulomb.method == COULOMB_MMM2D || coulomb.method == COULOMB_MMM1D)) {
    runtimeErrorMsg() << "ICCP3M implemented only for MMM1D,MMM2D,ELC or P3M ";
  }
  switch (coulomb.method) {
#ifdef P3M
  case COULOMB_ELC_P3M:
    // if (elc_params.dielectric_contrast_on) {
    //   runtimeErrorMsg() << "ICCP3M conflicts with ELC dielectric constrast";
    // }
    p3m_charge_assign();
    p3m_calc_kspace_forces(1, 0);
    ELC_add_force();
    break;

#ifdef CUDA
  case COULOMB_P3M_GPU:
    if (this_node == 0) {
      FORCE_TRACE(printf("Computing GPU P3M forces.\n"));
      p3m_gpu_add_farfield_force();
    }
    break;
#endif
  case COULOMB_P3M:
    p3m_charge_assign();
    p3m_calc_kspace_forces(1, 0);
    break;
#endif
  case COULOMB_MMM2D:
    MMM2D_add_far_force();
    MMM2D_dielectric_layers_force_contribution();
    break;
  default:
    break;
  }

#endif
}

void c_splitParticles(PartCfg &partCfg, bool force) {
  for (auto &p : partCfg) {
    auto const id = p.p.identity - iccp3m_cfg.first_id;
    if (id < iccp3m_cfg.n_icc + iccp3m_cfg.numMissingIDs &&
        id >= 0) {
          // check if particle is splittable
          iccShape * shapePointer = iccp3m_data.iccTypes[p.adapICC.iccTypeID];
          // fprintf(stderr, "%d\n", p.adapICC.iccTypeID);
          if (shapePointer->cutoff[0] <= p.adapICC.displace[0] &&
              shapePointer->cutoff[1] <= p.adapICC.displace[1] &&
              shapePointer->cutoff[2] <= p.adapICC.displace[2]) {

                if (force || (std::abs(p.p.q) > iccp3m_cfg.maxCharge)) {
                  // split this particle
                  shapePointer->splitExt(p, iccp3m_data.newParticleData);
                }
          }
    }
  }
}

void c_reduceParticle() {
  iccp3m_data.iccTypes[iccp3m_data.reducedPart.iccTypeID]->reduceExt(iccp3m_data.reducedPart);
  // this call might not be necessary! -> direct call from cython
}

void c_checkSet(int ID) {
  auto it = iccp3m_data.missingIDs.end();
  while (it != iccp3m_data.missingIDs.begin()) {
    if (*(--it) == ID) {
      iccp3m_data.missingIDs.erase(it);
      iccp3m_data.largestID--;
      ID--;
    } else {
      break;
    }
  }
}

int c_addTypeWall(Vector3d cutoff, bool useTrans, double transMatrix[9], double invMatrix[9]) {
  iccWall * wall = new iccWall(cutoff, useTrans, &transMatrix[0], &invMatrix[0]);
  iccp3m_data.iccTypes.push_back(wall);
  return iccp3m_data.iccTypes.size();
}

int c_addTypeCylinder(Vector3d center, Vector3d axis, double length, double radius, double direction, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix) {
  iccCylinder * cylinder = new iccCylinder(center, axis, length, radius, direction, cutoff, useTrans, transMatrix, invMatrix);
  iccp3m_data.iccTypes.push_back(cylinder);
  return iccp3m_data.iccTypes.size();
}

int c_addTypeTorus(Vector3d center, Vector3d axis, double length, double radius, double smoothingRadius, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix) {
  iccTorus * torus = new iccTorus(center, axis, length, radius, smoothingRadius, cutoff, useTrans, transMatrix, invMatrix);
  iccp3m_data.iccTypes.push_back(torus);
  return iccp3m_data.iccTypes.size();
}

int c_addTypeInterface(Vector3d center, double radius, double smoothingRadius, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix) {
  iccInterface * interface = new iccInterface(center, radius, smoothingRadius, cutoff, useTrans, transMatrix, invMatrix);
  iccp3m_data.iccTypes.push_back(interface);
  return iccp3m_data.iccTypes.size();
}

int c_outputVTK(char * filename, PartCfg & partCfg) {
  FILE *fp = fopen(filename, "w");

  if (fp == nullptr) {
    return 1;
  }

  std::vector<Vector3d> forces;
  forces.reserve(iccp3m_cfg.n_icc);

  std::vector<double> charges;
  charges.reserve(iccp3m_cfg.n_icc);

  fprintf(fp, "\
# vtk DataFile Version 3.0\n\
vtk output\n\
ASCII\n\
DATASET POLYDATA\n\
POINTS %u double\n", iccp3m_cfg.n_icc);

  for (auto &p : partCfg) {
    auto const id = p.p.identity - iccp3m_cfg.first_id;
    if (id < iccp3m_cfg.n_icc + iccp3m_cfg.numMissingIDs &&
        id >= 0) {
          auto pp = folded_position(p);
          fprintf(fp, "%f %f %f ", pp[0], pp[1], pp[2]);
          forces.push_back(p.f.f / p.p.q);
          charges.push_back(p.p.q);
    }
  }

  fprintf(fp, "\n\
POINT_DATA %u\n\
SCALARS charge double\n\
LOOKUP_TABLE default\n", iccp3m_cfg.n_icc);

  for (auto & charge : charges) {
    fprintf(fp, "%f ", charge);
  }

  fprintf(fp, "\n\
VECTORS field double\n");

  for (auto & force : forces) {
    fprintf(fp, "%f %f %f ", force[0], force[1], force[2]);
  }

  fclose(fp);

  return 0;
}

int c_outputParticle(char * filename, PartCfg & partCfg) {
  FILE *fp = fopen(filename, "w");

  if (fp == nullptr) {
    return 1;
  }

  std::vector<Vector3d> forces;
  forces.reserve(iccp3m_cfg.first_id);

  fprintf(fp, "\
# vtk DataFile Version 3.0\n\
vtk output\n\
ASCII\n\
DATASET POLYDATA\n\
POINTS %u double\n", iccp3m_cfg.first_id);

  // position
  for (auto &p : partCfg) {
    if (p.p.identity < iccp3m_cfg.first_id) {
          auto pp = folded_position(p);
          fprintf(fp, "%f %f %f ", pp[0], pp[1], pp[2]);
          forces.push_back(p.f.f);
    }
  }

  fprintf(fp, "\n\
POINT_DATA %u\n\
VECTORS field double\n", iccp3m_cfg.first_id);

  for (auto & force : forces) {
    fprintf(fp, "%f %f %f ", force[0], force[1], force[2]);
  }
  fclose(fp);

  return 0;
}




/** \name Private Functions */
/************************************************************/
/*@{*/

int iccp3m_sanity_check() {
  switch (coulomb.method) {
#ifdef P3M
  case COULOMB_ELC_P3M: {
    // if (elc_params.dielectric_contrast_on) {
    //   runtimeErrorMsg() << "ICCP3M conflicts with ELC dielectric constrast";
    //   return 1;
    // }
    break;
  }
#endif
  case COULOMB_DH: {
    runtimeErrorMsg() << "ICCP3M does not work with Debye-Hueckel iccp3m.h";
    return 1;
  }
  case COULOMB_RF: {
    runtimeErrorMsg() << "ICCP3M does not work with COULOMB_RF iccp3m.h";
    return 1;
  }
  default:
    break;
  }

#ifdef NPT
  if (integ_switch == INTEG_METHOD_NPT_ISO) {
    runtimeErrorMsg() << "ICCP3M does not work in the NPT ensemble";
    return 1;
  }
#endif

  return 0;
}

#endif
