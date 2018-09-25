#
# Copyright (C) 2013-2018 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import print_function, absolute_import
from . cimport utils
include "myconfig.pxi"
from espressomd cimport actors
from . import actors
import numpy as np
from espressomd.utils cimport handle_errors
from . import utils
from .system import System

IF ELECTROSTATICS and P3M:
    from espressomd.electrostatics import check_neutrality

    cdef class ElectrostaticExtensions(actors.Actor):
        pass

    cdef class ELC(ElectrostaticExtensions):
        """
        Electrostatics solver for systems with two periodic dimensions.

        Parameters
        ----------
        gap_size                : float, required
                                  The gap size gives the height of the empty region between the system box
                                  and the neighboring artificial images. |es| does not
                                  make sure that the gap is actually empty, this is the users
                                  responsibility. The method will compute fine if the condition is not
                                  fulfilled, however, the error bound will not be reached. Therefore you
                                  should really make sure that the gap region is empty (e.g. with wall
                                  constraints).
        maxPWerror              : float, required
                                  The maximal pairwise error sets the least upper bound (LUB) error of
                                  the force between any two charges without prefactors (see the papers).
                                  The algorithm tries to find parameters to meet this LUB requirements or
                                  will throw an error if there are none.
        delta_mid_top           : float, optional
                                  This parameter sets the dielectric contrast
                                  between the upper boundary and the simulation
                                  box :math:`\\Delta_t`.
        delta_mid_bottom        : float, optional
                                  This parameter sets the dielectric contrast
                                  between the lower boundary and the simulation
                                  box :math:`\\Delta_b`.
        const_pot               : int, optional
                                  Selector parameter for setting a constant
                                  electric potential between the top and bottom
                                  of the simulation box.
        pot_diff                : float, optional
                                  If const_pot mode is selected this parameter
                                  controls the applied voltage.
        neutralize              : int, optional
                                  By default, ELC just as P3M adds a homogeneous neutralizing background
                                  to the system in case of a net charge. However, unlike in three dimensions,
                                  this background adds a parabolic potential across the
                                  slab :cite:`ballenegger09a`. Therefore, under normal circumstance, you will
                                  probably want to disable the neutralization for non-neutral systems.
                                  This corresponds then to a formal regularization of the forces and
                                  energies :cite:`ballenegger09a`. Also, if you add neutralizing walls
                                  explicitly as constraints, you have to disable the neutralization.
                                  When using a dielectric contrast or full metallic walls (`delta_mid_top
                                  != 0` or `delta_mid_bot != 0` or `const_pot_on=1`), `neutralize` is
                                  overwritten and switched off internally. Note that the special case of
                                  non-neutral systems with a *non-metallic* dielectric jump (eg.
                                  `delta_mid_top` or `delta_mid_bot` in `]-1,1[`) is not covered by the
                                  algorithm and will throw an error.
        far_cut                 : float, optional
                                  Cut off radius, use with care, intended for testing purposes.
        """

        def validate_params(self):
            default_params = self.default_params()
            check_type_or_throw_except(
                self._params["maxPWerror"], 1, float, "")
            check_range_or_except(
                self._params, "maxPWerror", 0, False, "inf", True)
            check_type_or_throw_except(self._params["gap_size"], 1, float, "")
            check_range_or_except(
                self._params, "gap_size", 0, False, "inf", True)
            check_type_or_throw_except(self._params["far_cut"], 1, float, "")
            check_type_or_throw_except(
                self._params["neutralize"], 1, type(True), "")

        def valid_keys(self):
            return "maxPWerror", "gap_size", "far_cut", "neutralize", "delta_mid_top", "delta_mid_bot", "const_pot", "pot_diff", "check_neutrality"

        def required_keys(self):
            return ["maxPWerror", "gap_size"]

        def default_params(self):
            return {"maxPWerror": -1,
                    "gap_size": -1,
                    "far_cut": -1,
                    "delta_mid_top": 0,
                    "delta_mid_bot": 0,
                    "const_pot": 0,
                    "pot_diff": 0.0,
                    "neutralize": True,
                    "check_neutrality": True}

        def _get_params_from_es_core(self):
            params = {}
            params.update(elc_params)
            return params

        def _set_params_in_es_core(self):
            if coulomb.method == COULOMB_P3M_GPU:
                raise Exception(
                    "ELC tuning failed, ELC is not set up to work with the GPU P3M")

            if self._params["const_pot"]:
                self._params["delta_mid_top"] = -1
                self._params["delta_mid_bot"] = -1

            if ELC_set_params(
                self._params["maxPWerror"],
                self._params["gap_size"],
                self._params["far_cut"],
                int(self._params["neutralize"]),
                self._params["delta_mid_top"],
                self._params["delta_mid_bot"],
                int(self._params["const_pot"]),
                    self._params["pot_diff"]):
                handle_errors(
                    "ELC tuning failed, ELC is not set up to work with the GPU P3M")

        def _activate_method(self):
            check_neutrality(self._params)
            self._set_params_in_es_core()

        def _deactivate_method(self):
            raise Exception(
                "Unable to remove ELC as the state of the underlying electrostatics method will remain unclear.")

    cdef class ICC(ElectrostaticExtensions):
        """
        Interface to the induced charge calculation scheme for dielectric interfaces

        See :ref:`Dielectric interfaces with the ICC algorithm`

        """

        def validate_params(self):
            default_params = self.default_params()

            check_type_or_throw_except(self._params["n_icc"], 1, int, "")
            check_range_or_except(
                self._params, "n_icc", 1, True, "inf", True)

            check_type_or_throw_except(
                self._params["convergence"], 1, float, "")
            check_range_or_except(
                self._params, "convergence", 0, False, "inf", True)

            check_type_or_throw_except(
                self._params["relaxation"], 1, float, "")
            check_range_or_except(
                self._params, "relaxation", 0, False, "inf", True)

            check_type_or_throw_except(
                self._params["ext_field"], 3, float, "")

            check_type_or_throw_except(
                self._params["max_iterations"], 1, int, "")
            check_range_or_except(
                self._params, "max_iterations", 0, False, "inf", True)

            check_type_or_throw_except(
                self._params["first_id"], 1, int, "")
            check_range_or_except(
                self._params, "first_id", 0, True, "inf", True)

            check_type_or_throw_except(
                self._params["eps_out"], 1, float, "")

            check_type_or_throw_except(
                    self._params["maxCharge"], 1, float, "")
            check_range_or_except(
                self._params, "maxCharge", 0, True, "inf", True)

            check_type_or_throw_except(
                    self._params["minCharge"], 1, float, "")
            check_range_or_except(
                self._params, "minCharge", 0, True, "inf", True)

            # Required list input
            self._params["normals"] = np.array(self._params["normals"])
            if self._params["normals"].size != self._params["n_icc"] * 3:
                raise ValueError(
                    "Expecting normal list with " + self._params["n_icc"] * 3 + " entries.")
            check_type_or_throw_except(self._params["normals"], self._params[
                "n_icc"], np.ndarray, "Error in normal list.")

            check_type_or_throw_except(
                self._params["areas"], self._params["n_icc"], float, "Error in area list.")

            # Not Required
            if "sigmas" in self._params.keys():
                check_type_or_throw_except(
                    self._params["sigmas"], self._params["n_icc"], float, "Error in sigma list.")
            else:
                self._params["sigmas"] = np.zeros(self._params["n_icc"])

            if "epsilons" in self._params.keys():
                check_type_or_throw_except(
                    self._params["epsilons"], self._params["n_icc"], float, "Error in epsilon list.")
            else:
                self._params["epsilons"] = np.zeros(self._params["n_icc"])

            if not ("maxCharge" in self._params.keys() and "minCharge" in self._params.keys()):
                self._params["maxCharge"] = 0.
                self._params["minCharge"] = 0.

        def valid_keys(self):
            return "n_icc", "convergence", "relaxation", "ext_field", "max_iterations", "first_id", "eps_out", "normals", "areas", "sigmas", "epsilons", "check_neutrality", "maxCharge", "minCharge"

        def required_keys(self):
            return [ "n_icc", "normals", "areas"]

        def default_params(self):
            return {"n_icc": 0,
                    "convergence": 1e-3,
                    "relaxation": 0.7,
                    "ext_field": [0, 0, 0],
                    "max_iterations": 100,
                    "first_id": 0,
                    "eps_out": 1,
                    "normals": [],
                    "areas": [],
                    "sigmas": [],
                    "epsilons": [],
                    "check_neutrality": True,
                    "maxCharge": 0.,
                    "minCharge": 0.}

        def _get_params_from_es_core(self):
            params = {}
            params["n_icc"] = iccp3m_cfg.n_ic

            # Fill Lists
            normals = []
            areas = []
            sigmas = []
            epsilons = []
            for i in range(iccp3m_cfg.n_ic):
                normals.append([iccp3m_cfg.normals[i][0], iccp3m_cfg.normals[
                               i][1], iccp3m_cfg.normals[i][2]])
                areas.append(iccp3m_cfg.areas[i])
                epsilons.append(iccp3m_cfg.ein[i])
                sigmas.append(iccp3m_cfg.sigma[i])

            params["normals"] = normals
            params["areas"] = areas
            params["epsilons"] = epsilons
            params["sigmas"] = sigmas

            params["ext_field"] = [iccp3m_cfg.ext_field[0],
                                   iccp3m_cfg.ext_field[1], iccp3m_cfg.ext_field[2]]
            params["first_id"] = iccp3m_cfg.first_id
            params["max_iterations"] = iccp3m_cfg.num_iteration
            params["convergence"] = iccp3m_cfg.convergence
            params["relaxation"] = iccp3m_cfg.relax
            params["eps_out"] = iccp3m_cfg.eout

            params["maxCharge"] = iccp3m_data.maxCharge
            params["minCharge"] = iccp3m_data.minCharge

            return params

        def _set_params_in_es_core(self):
            # First set number of icc particles
            iccp3m_cfg.n_ic = self._params["n_icc"]
            iccp3m_cfg.numMissingIDs = iccp3m_data.missingIDs.size();
            # Allocate ICC lists
            iccp3m_alloc_lists()

            # Fill Lists
            for i in range(iccp3m_cfg.n_ic):
                iccp3m_cfg.normals[i][0] = self._params["normals"][i][0]
                iccp3m_cfg.normals[i][1] = self._params["normals"][i][1]
                iccp3m_cfg.normals[i][2] = self._params["normals"][i][2]

                iccp3m_cfg.areas[i] = self._params["areas"][i]
                iccp3m_cfg.ein[i] = self._params["epsilons"][i]
                iccp3m_cfg.sigma[i] = self._params["sigmas"][i]

            iccp3m_cfg.ext_field[0] = self._params["ext_field"][0]
            iccp3m_cfg.ext_field[1] = self._params["ext_field"][1]
            iccp3m_cfg.ext_field[2] = self._params["ext_field"][2]
            iccp3m_cfg.first_id = self._params["first_id"]
            iccp3m_cfg.num_iteration = self._params["max_iterations"]
            iccp3m_cfg.convergence = self._params["convergence"]
            iccp3m_cfg.relax = self._params["relaxation"]
            iccp3m_cfg.eout = self._params["eps_out"]
            iccp3m_cfg.citeration = 0
            iccp3m_data.maxCharge = self._params["maxCharge"]
            iccp3m_data.minCharge = self._params["minCharge"]
            iccp3m_cfg.largestID = self._params["first_id"] + self._params["n_icc"]

            # Broadcasts vars
            mpi_iccp3m_init()

        def _addNewParticlesToSystem(self, _system):
            rerun = False
            cdef vector[int] currID
            # loop over all particles to split
            counter = 0
            while (iccp3m_data.newParticleData.size() != 0):
                frontData = iccp3m_data.newParticleData.front()
                counter += 1
                rerun = True
                currID.clear()
                # first particle is only modified
                currID.push_back(frontData[0].parentID)

                print(frontData[0].charge)
                print(frontData[0].area)
                print(frontData[0].normal[0], frontData[0].normal[1], frontData[0].normal[2])

                parentPart = _system.part[frontData[0].parentID]
                parentPart.pos = [frontData[0].pos[0],
                                  frontData[0].pos[1],
                                  frontData[0].pos[2]]
                parentPart.q = frontData[0].charge
                parentPart.displace = [frontData[0].displace[0],
                                       frontData[0].displace[1],
                                       frontData[0].displace[2]]
                parentPart.normal = [frontData[0].normal[0],
                                     frontData[0].normal[1],
                                     frontData[0].normal[2]]
                parentPart.area = frontData[0].area

                # add leftover particles
                for i in range(1, frontData.size()):
                    print(frontData[i].charge)
                    print(frontData[i].area)
                    print(frontData[i].normal[0], frontData[i].normal[1], frontData[i].normal[2])
                    currID.push_back(iccp3m_cfg.largestID)
                    _system.part.add(id=iccp3m_cfg.largestID,
                                     pos=[frontData[i].pos[0],
                                          frontData[i].pos[1],
                                          frontData[i].pos[2]],
                                     q=frontData[i].charge,
                                     area=frontData[i].area,
                                     eps=frontData[i].eps,
                                     sigma=frontData[i].sigma,
                                     displace=[frontData[i].displace[0],
                                              frontData[i].displace[1],
                                              frontData[i].displace[2]],
                                     normal=[frontData[i].normal[0],
                                             frontData[i].normal[1],
                                             frontData[i].normal[2]],
                                     iccTypeID=frontData[i].iccTypeID,
                                     type=frontData[i].typeID,
                                     fix=[1, 1, 1])
                    iccp3m_cfg.largestID += 1
                    iccp3m_cfg.n_ic += 1
                # remove vector from list
                iccp3m_data.newParticleData.pop()
                iccp3m_data.trackList.push_back(currID)
            return rerun


        def _activate_method(self):
            check_neutrality(self._params)
            self._set_params_in_es_core()

        def _deactivate_method(self):
            iccp3m_cfg.n_ic = 0
            # Allocate ICC lists
            iccp3m_alloc_lists()

            # Broadcasts vars
            mpi_iccp3m_init()

        def rebuildData(self):
            iccp3m_cfg.numMissingIDs = iccp3m_data.missingIDs.size()
            iccp3m_alloc_lists()
            c_rebuildData(partCfg())

            mpi_iccp3m_init()

        def splitParticles(self, _system, _rerun=True, _force=False):
            c_splitParticles(partCfg(), _force)

            if self._addNewParticlesToSystem(_system):
                self.rebuildData()
                if _rerun:
                    self.splitParticles(_system, _rerun=True, _force=_force)

        def outputCharges(self):
            c_getCharges(partCfg())
            return iccp3m_data.iccCharges

        def newParticles(self, _number):
            iccp3m_cfg.n_ic = _number


        def reduceParticles(self, _system, _force=False):
            cdef set[int] noReduce
            skip = False
            summe = 0.;

            c_getCharges(partCfg())
            for vec in reversed(iccp3m_data.trackList):
                skip = False

                # check if any particle is unreducable
                if not any(noReduce.count(vec[i]) for i in range(len(vec))):
                    # calculate charge sum
                    for i in range(len(vec)):
                        summe += iccp3m_data.iccCharges[i]
                    if _force or abs(summe) < iccp3m_data.minCharge:

                        iccp3m_data.reducedPart.iccTypeID = _system.part[vec[0]].iccTypeID
                        for i in range(3):
                            iccp3m_data.reducedPart.pos[i] = _system.part[vec[0]].pos[i]
                            iccp3m_data.reducedPart.displace[i] = _system.part[vec[0]].displace[i]
                            iccp3m_data.reducedPart.normal[i] = _system.part[vec[0]].normal[i]

                        c_reduceParticle()

                        # delete other particles
                        # update numMissingIDs if not the last ones <- problem here!

                        part = _system.part[vec[0]]
                        part.pos = [iccp3m_data.reducedPart.pos[0],
                                    iccp3m_data.reducedPart.pos[1],
                                    iccp3m_data.reducedPart.pos[2]]
                        part.normal = [iccp3m_data.reducedPart.normal[0],
                                       iccp3m_data.reducedPart.normal[1],
                                       iccp3m_data.reducedPart.normal[2]]
                        part.area = sum(_system.part[vec].area)
                        part.q = sum(_system.part[vec].q)
                        part.displace = [iccp3m_data.reducedPart.displace[0],
                                         iccp3m_data.reducedPart.displace[1],
                                         iccp3m_data.reducedPart.displace[2]]

                        iccp3m_cfg.n_ic -= len(vec) - 1
                        if vec[-1] == (iccp3m_cfg.largestID - 1):
                            iccp3m_cfg.largestID -= len(vec) - 1
                            c_checkSet(vec[1] - 1)
                        else:
                            for i in range(1, len(vec)):
                                iccp3m_data.missingIDs.insert(vec[i])

                        # print('{} - {}'.format(vec, len(_system.part)))
                        _system.part[vec[1]:vec[-1]+1].remove()
                        iccp3m_data.trackList.remove(vec)
                else:
                    print('actually got to else!')
                    noReduce.insert(vec)
                    # exit early if all particles are unsplittable
                    if len(noReduce) >= iccp3m_cfg.n_ic:
                        print('early breakout!')
                        break
            iccp3m_cfg.numMissingIDs = iccp3m_data.missingIDs.size()
            c_rebuildData(partCfg())

        def outputICCData(self, _filename):
            if (c_outputVTK(utils.to_char_pointer(_filename), partCfg())):
                print('Something seemed to went wrong!')

        def outputParticleData(self, _filename):
            if (c_outputParticle(utils.to_char_pointer(_filename), partCfg())):
                print('Something seemed to went wrong!')

        def addTypeWall(self, _cutoff, _useTrans=False, _transMatrix=None, _invMatrix=None):
            cdef Vector3d cutoff
            cdef bool useTrans
            cdef double transMatrix[9]
            cdef double invMatrix[9]

            check_type_or_throw_except(_cutoff, 3, float, "cutoff has to be floats")
            check_type_or_throw_except(_useTrans, 1, int, "useTrans has to be integer")
            for i in range(3):
                cutoff[i] = _cutoff[i]

            if _transMatrix is not None and _invMatrix is not None:
                check_type_or_throw_except(_transMatrix, 9, float, "Matrix has to be 9 floats")
                check_type_or_throw_except(_invMatrix, 9, float, "Matrix has to be 9 floats")
                for i in range(9):
                    transMatrix[i] = _transMatrix[i]
                    invMatrix[i] = _invMatrix[i]

            useTrans = _useTrans

            return c_addTypeWall(cutoff, useTrans, transMatrix, invMatrix)

        def addTypeCylinder(self, _center, _axis, _length, _radius, _direction, _cutoff, _useTrans=False, _transMatrix=None, _invMatrix=None):
            cdef Vector3d center
            cdef Vector3d axis
            cdef Vector3d cutoff
            cdef double length
            cdef double radius
            cdef double direction
            cdef bool useTrans
            cdef double transMatrix[9]
            cdef double invMatrix[9]

            check_type_or_throw_except(_center, 3, float, "center has to be floats")
            check_type_or_throw_except(_axis, 3, float, "axis has to be floats")
            check_type_or_throw_except(_length, 1, float, "length has to be float")
            check_type_or_throw_except(_radius, 1, float, "radius has to be float")
            # might consider {-1, 1}
            check_type_or_throw_except(_direction, 1, float, "direction has to be float")
            check_type_or_throw_except(_cutoff, 3, float, "cutoff has to be floats")
            check_type_or_throw_except(_useTrans, 1, int, "useTrans has to be integer")

            for i in range(3):
                center[i] = _center[i]
                axis[i] = _axis[i]
                cutoff[i] = _cutoff[i]
            length = _length
            radius = _radius
            direction = _direction

            if _transMatrix is not None and _invMatrix is not None:
                check_type_or_throw_except(_transMatrix, 9, float, "Matrix has to be 9 floats")
                check_type_or_throw_except(_invMatrix, 9, float, "Matrix has to be 9 floats")
                for i in range(9):
                    transMatrix[i] = _transMatrix[i]
                    invMatrix[i] = _invMatrix[i]

            useTrans = _useTrans

            return c_addTypeCylinder(center, axis, length, radius, direction, cutoff, useTrans, transMatrix, invMatrix)
        def addTypeTorus(self, _center, _axis, _length, _radius, _smoothingRadius, _cutoff, _useTrans=False, _transMatrix=None, _invMatrix=None):
            cdef Vector3d center
            cdef Vector3d axis
            cdef Vector3d cutoff
            cdef double length
            cdef double radius
            cdef double smoothingRadius
            cdef bool useTrans
            cdef double transMatrix[9]
            cdef double invMatrix[9]

            check_type_or_throw_except(_center, 3, float, "center has to be floats")
            check_type_or_throw_except(_axis, 3, float, "axis has to be floats")
            check_type_or_throw_except(_length, 1, float, "length has to be float")
            check_type_or_throw_except(_radius, 1, float, "radius has to be float")
            check_type_or_throw_except(_smoothingRadius, 1, float, "smoothingRadius has to be float")
            check_type_or_throw_except(_cutoff, 3, float, "cutoff has to be floats")
            check_type_or_throw_except(_useTrans, 1, int, "useTrans has to be integer")

            for i in range(3):
              center[i] = _center[i]
              axis[i] = _axis[i]
              cutoff[i] = _cutoff[i]
            length = _length
            radius = _radius
            smoothingRadius = _smoothingRadius

            if _transMatrix is not None and _invMatrix is not None:
              check_type_or_throw_except(_transMatrix, 9, float, "Matrix has to be 9 floats")
              check_type_or_throw_except(_invMatrix, 9, float, "Matrix has to be 9 floats")
              for i in range(9):
                  transMatrix[i] = _transMatrix[i]
                  invMatrix[i] = _invMatrix[i]

            useTrans = _useTrans

            return c_addTypeTorus(center, axis, length, radius, smoothingRadius, cutoff, useTrans, transMatrix, invMatrix)

        def addTypeInterface(self,_center, _radius, _smoothingRadius, _cutoff, _useTrans=False, _transMatrix=None, _invMatrix=None):
            cdef Vector3d center
            cdef Vector3d cutoff
            cdef double radius
            cdef double smoothingRadius
            cdef bool useTrans
            cdef double transMatrix[9]
            cdef double invMatrix[9]

            check_type_or_throw_except(_center, 3, float, "center has to be floats")
            check_type_or_throw_except(_radius, 1, float, "radius has to be float")
            check_type_or_throw_except(_smoothingRadius, 1, float, "smoothingRadius has to be float")
            check_type_or_throw_except(_cutoff, 3, float, "cutoff has to be floats")
            check_type_or_throw_except(_useTrans, 1, int, "useTrans has to be integer")

            for i in range(3):
              center[i] = _center[i]
              cutoff[i] = _cutoff[i]
            radius = _radius
            smoothingRadius = _smoothingRadius

            if _transMatrix is not None and _invMatrix is not None:
              check_type_or_throw_except(_transMatrix, 9, float, "Matrix has to be 9 floats")
              check_type_or_throw_except(_invMatrix, 9, float, "Matrix has to be 9 floats")
              for i in range(9):
                  transMatrix[i] = _transMatrix[i]
                  invMatrix[i] = _invMatrix[i]

            useTrans = _useTrans

            return c_addTypeInterface(center, radius, smoothingRadius, cutoff, useTrans, transMatrix, invMatrix)
