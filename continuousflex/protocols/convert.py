# **************************************************************************
# *
# * Authors:
# * Mohamad Harastani (mohamad.harastani@igbmc.fr)
# * J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# * Slavica Jonic (slavica.jonic@upmc.fr)
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
# *
# **************************************************************************

from collections import OrderedDict
from pwem.emlib import (MDL_NMA_MODEFILE, MDL_NMA_COLLECTIVITY, MDL_NMA_SCORE, MDL_ORDER)
from pyworkflow.utils import Environ
from pwem.objects import NormalMode

from xmipp3.convert import rowToObject, objectToRow
from continuousflex.constants import NMA_HOME
import numpy as np
import math

MODE_DICT = OrderedDict([
    ("_modeFile", MDL_NMA_MODEFILE),
    ("_collectivity", MDL_NMA_COLLECTIVITY),
    ("_score", MDL_NMA_SCORE),
])

"""
Represents a point within a multidimensional dataset together with
its associated metadata, including spatial coordinates, weight, and
selection status. The class serves as the fundamental element for
managing geometric, statistical, or visualization-oriented data in
interactive analysis workflows.

AI Generated:

Point and Data Management (Point) - User Manual
    Overview

    The Point protocol provides a framework for representing,
    organizing, and manipulating collections of multidimensional
    points. Its primary purpose is to support analytical and
    visualization workflows in which individual observations are
    associated with spatial coordinates and additional descriptive
    properties.

    A point is more than a simple coordinate. In addition to its
    position, each element may carry a numerical importance value
    and a logical state describing whether it is active, selected,
    or excluded from analysis. This design allows the same dataset
    to support exploration, filtering, annotation, and interactive
    selection without permanently modifying the original data.

    Data Organization

    The framework manages collections of points as coherent datasets.
    Each dataset preserves the relationship between individual
    elements while providing convenient access to coordinate values,
    weights, and selection information. This organization is useful
    for applications involving dimensionality reduction, clustering,
    trajectory analysis, geometric measurements, or interactive
    plotting environments.

    Datasets may contain two-dimensional or three-dimensional
    coordinates as well as additional numerical descriptors. The
    coordinate system remains consistent across all points,
    facilitating comparison and interpretation of spatial patterns.

    Selection and Filtering

    A central feature of the framework is the ability to distinguish
    between active, selected, and discarded elements. Selected
    points can be used to define regions of interest, identify
    representative observations, or support manual curation.
    Discarded points remain stored within the dataset but are
    excluded from standard analysis operations.

    This approach allows users to explore alternative selections
    without losing information. As a result, workflows remain
    flexible and reversible throughout the analysis process.

    Mathematical Exploration

    The framework supports evaluation of mathematical relationships
    involving point-associated variables. This capability enables
    users to derive new measurements, explore custom metrics, or
    investigate relationships among dimensions without creating
    separate datasets.

    Such flexibility is particularly useful during exploratory
    analysis, where researchers often need to test hypotheses and
    evaluate different combinations of variables before deciding on
    a final interpretation.

    Path-Based Analysis

    In addition to general datasets, the framework supports ordered
    collections of points that define trajectories or paths through
    a coordinate space. These paths can represent motion,
    transitions between states, interpolation routes, or user-defined
    exploration trajectories.

    The path representation allows refinement of trajectories by
    introducing additional intermediate positions. This capability
    helps create smoother paths, improve sampling density, and
    support analyses that require continuous transitions between
    neighboring states.

    Outputs and Interpretation

    The resulting datasets provide structured access to coordinates,
    weights, and selection information while preserving the original
    relationships between points. Users can extract coordinate
    distributions, analyze subsets of interest, or construct
    trajectories for visualization and further computation.

    Because discarded and selected elements remain explicitly
    represented, the framework supports transparent and reproducible
    analysis decisions throughout the workflow.

    Practical Recommendations

    When working with exploratory datasets, it is often beneficial
    to use selection states to identify candidate regions of
    interest before performing more detailed analyses. Maintaining
    discarded elements within the dataset can also facilitate later
    reevaluation of filtering decisions.

    For trajectory-based studies, adding intermediate points may
    improve visual continuity and provide a more accurate
    representation of gradual transitions between neighboring
    states.

    Final Perspective

    The framework provides a flexible foundation for managing
    multidimensional point collections and ordered trajectories.
    By combining coordinate storage, state management, weighting,
    and path handling within a unified structure, it supports a
    broad range of visualization, exploration, and analytical
    workflows while preserving the integrity and interpretability
    of the underlying data.
"""


def rowToMode(row):
    """ Set properties of a NormalMode object from a Metadata row. """
    mode = NormalMode()
    rowToObject(row, mode, MODE_DICT)
    mode.setObjId(row.getValue(MDL_ORDER))
    return mode


def modeToRow(mode, row):
    """ Write the MetaData row from a given NormalMode object. """
    row.setValue(MDL_ORDER, int(mode.getObjId()))
    objectToRow(mode, row, MODE_DICT)


def getNMAEnviron():
    """ Create the needed environment for NMA programs. """
    from continuousflex import Plugin
    environ = Plugin.getEnviron()
    environ.update({'PATH': Plugin.getVar(NMA_HOME)}, position=Environ.BEGIN)
    environ.update({'LD_LIBRARY_PATH': Plugin.getCondaLibPath()}, position=Environ.BEGIN)
    return environ


def eulerAngles2matrix(alpha, beta, gamma, shiftx, shifty, shiftz):
    A = np.empty([4, 4])
    A.fill(2)
    A[3, 3] = 1
    A[3, 0:3] = 0
    A[0, 3] = float(shiftx)
    A[1, 3] = float(shifty)
    A[2, 3] = float(shiftz)
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    sa = np.sin(np.deg2rad(alpha))
    ca = np.cos(np.deg2rad(alpha))
    sb = np.sin(np.deg2rad(beta))
    cb = np.cos(np.deg2rad(beta))
    sg = np.sin(np.deg2rad(gamma))
    cg = np.cos(np.deg2rad(gamma))
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa
    A[0, 0] = cg * cc - sg * sa
    A[0, 1] = cg * cs + sg * ca
    A[0, 2] = -cg * sb
    A[1, 0] = -sg * cc - cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] = sg * sb
    A[2, 0] = sc
    A[2, 1] = ss
    A[2, 2] = cb
    return A


def matrix2eulerAngles(A):
    abs_sb = np.sqrt(A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2])
    if (abs_sb > 16 * np.exp(-5)):
        gamma = math.atan2(A[1, 2], -A[0, 2])
        alpha = math.atan2(A[2, 1], A[2, 0])
        if (abs(np.sin(gamma)) < np.exp(-5)):
            sign_sb = np.sign(-A[0, 2] / np.cos(gamma))
        else:
            if np.sin(gamma) > 0:
                sign_sb = np.sign(A[1, 2])
            else:
                sign_sb = -np.sign(A[1, 2])
        beta = math.atan2(sign_sb * abs_sb, A[2, 2])
    else:
        if (np.sign(A[2, 2]) > 0):
            alpha = 0
            beta = 0
            gamma = math.atan2(-A[1, 0], A[0, 0])
        else:
            alpha = 0
            beta = np.pi
            gamma = math.atan2(A[1, 0], -A[0, 0])
    gamma = np.rad2deg(gamma)
    beta = np.rad2deg(beta)
    alpha = np.rad2deg(alpha)
    return alpha, beta, gamma, A[0, 3], A[1, 3], A[2, 3]


def l2(Vec1, Vec2):
    Vec1 = np.array(Vec1)
    Vec2 = np.array(Vec2)
    value = np.inner(Vec1 - Vec2, Vec1 - Vec2)
    return np.sqrt(value)
