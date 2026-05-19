# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# **************************************************************************

from continuousflex.protocols.protocol_genesis import FlexProtGenesis, EMFIT_VOLUMES, SIMULATION_NMMD

class FlexProtMDTOMO(FlexProtGenesis):
    """
    Performs molecular dynamics flexible fitting of structural models into
    electron microscopy volumes using the GENESIS simulation framework. The
    protocol is intended to explore conformational variability and structural
    transitions while maintaining physically realistic molecular behavior,
    allowing experimental density information to guide the simulation toward
    biologically meaningful conformations.

    AI Generated:

    MDTOMO (FlexProtMDTOMO) - User Manual
        Overview

        MDTOMO is a molecular dynamics tomography protocol designed to combine
        structural modeling with electron microscopy information in a unified
        simulation environment. Its main objective is to refine and explore
        conformational states of macromolecules by integrating molecular
        dynamics simulations with volumetric experimental data. This approach
        helps bridge the gap between static structural models and the dynamic
        behavior that biological systems often exhibit in solution.

        In structural biology, many macromolecular assemblies undergo motions
        that are essential for their function. Experimental density maps may
        capture one or more of these states but often do not directly describe
        the pathways connecting them. MDTOMO provides a framework for studying
        these motions while preserving physically plausible molecular
        interactions throughout the simulation.

        Inputs and General Workflow

        The protocol typically starts from an existing structural model and
        combines it with electron microscopy volume information. The structural
        model serves as the initial representation of the system, while the
        volumetric data provide experimental guidance that influences the
        conformational evolution during the simulation.

        The workflow is particularly useful when the available structural model
        does not perfectly match the observed density or when the objective is
        to investigate structural heterogeneity. By allowing the structure to
        adapt under both physical and experimental constraints, the protocol
        can reveal conformations that better explain the observed data.

        Molecular Dynamics and Flexible Fitting

        Unlike rigid fitting approaches, MDTOMO allows continuous structural
        deformation throughout the simulation. This capability is important for
        systems that experience domain movements, hinge motions, subunit
        rearrangements, or other large-scale conformational transitions.

        The molecular dynamics engine provides a physically motivated framework
        in which atomic interactions, structural restraints, and experimental
        information are balanced. As a result, the generated conformations are
        generally more realistic than those obtained through purely geometric
        fitting procedures.

        Use in Cryo-EM and Tomography

        MDTOMO is particularly valuable for cryo-EM and cryo-electron
        tomography studies where structural flexibility plays a significant
        role. Experimental maps often contain regions of varying resolution or
        represent ensembles of related conformations. Flexible fitting can help
        interpret these data by identifying structural arrangements that are
        consistent with the observed densities.

        For large molecular assemblies, membrane proteins, and dynamic
        complexes, the protocol can provide insight into motions that may be
        difficult to infer from static reconstructions alone.

        Outputs and Their Interpretation

        The primary outputs consist of molecular conformations generated during
        the simulation and refined against the experimental information.
        Depending on the biological system, these results may represent
        improved structural fits, alternative conformational states, or
        trajectories describing transitions between states.

        Interpretation should focus on biologically meaningful motions and on
        consistency with available experimental evidence. Structural changes
        that repeatedly appear during the simulation may indicate functionally
        relevant flexibility, although independent validation is always
        recommended.

        Practical Recommendations

        MDTOMO is most effective when the initial structural model already
        captures the overall architecture of the biological assembly. Large
        discrepancies between the starting model and the experimental density
        may require additional preprocessing or intermediate refinement steps.

        Users should evaluate simulation results together with biochemical,
        structural, and functional knowledge of the system. Flexible fitting
        can reveal plausible motions, but the biological significance of those
        motions should be assessed within the broader experimental context.

        Final Perspective

        MDTOMO provides a powerful strategy for integrating molecular dynamics
        simulations with electron microscopy volume data. By combining physical
        realism with experimental guidance, it enables the investigation of
        structural flexibility, conformational variability, and dynamic
        biological processes that are often inaccessible through static
        structural analysis alone.
    """
    _label = 'MDTOMO'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        FlexProtGenesis._defineParams(self, form)
        param = form.getParam("simulationType")
        param.setDefault(SIMULATION_NMMD)
        param = form.getParam("n_steps")
        param.setDefault(50000)
        param = form.getParam("EMfitChoice")
        param.setDefault(EMFIT_VOLUMES)
