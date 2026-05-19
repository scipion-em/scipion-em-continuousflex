# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

"""
Define some classes to store Data points for clustering.
"""


class Point:
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
    # Selection states
    DISCARDED = -1
    NORMAL = 0
    SELECTED = 1
    
    def __init__(self, pointId, data, weight, state=0):
        self._id = pointId
        self._data = data
        self._weight = weight
        self._state = state
        self._container = None
        
    def getId(self):
        return self._id
    
    def getX(self):
        return self._data[self._container.XIND]
    
    def setX(self, value):
        self._data[self._container.XIND] = value
    
    def getY(self):
        return self._data[self._container.YIND]
    
    def setY(self, value):
        self._data[self._container.YIND] = value
        
    def getZ(self):
        return self._data[self._container.ZIND]
        
    def setZ(self, value):
        self._data[self._container.ZIND] = value    
    
    def getWeight(self):
        return self._weight
    
    def getState(self):
        return self._state
    
    def setState(self, newState):
        self._state = newState
        
    def eval(self, expression):
        localDict = {}
        for i, x in enumerate(self._data):
            localDict['x%d' % (i+1)] = x
        return eval(expression, {"__builtins__":None}, localDict)

    def setSelected(self):
        self.setState(Point.SELECTED)
        
    def isSelected(self):
        return self.getState()==Point.SELECTED    
    
    def setDiscarded(self):
        self.setState(Point.DISCARDED)
           
    def isDiscarded(self):
        return self.getState()==Point.DISCARDED
    
    def getData(self):
        return self._data 

    
class Data():
    """ Store data points. """
    def __init__(self, **kwargs):
        # Indexes of data
        self._dim = kwargs.get('dim') # The points dimensions
        self.clear()
        
    def addPoint(self, point, position=None):
        point._container = self
        if position is None:
            self._points.append(point)
        else:
            self._points.insert(position, point)
            
    def getPoint(self, index):
        return self._points[index]
        
    def __iter__(self):
        for point in self._points:
            if not point.isDiscarded():
                yield point
                
    def iterAll(self):
        """ Iterate over all points, including the discarded ones."""
        return iter(self._points)
            
    def getXData(self):
        return [p.getX() for p in self]
    
    def getYData(self):
        return [p.getY() for p in self]
    
    def getZData(self):
        return [p.getZ() for p in self]
    
    def getWeights(self):
        return [p.getWeight() for p in self]
    
    def getSize(self):
        return len(self._points)
    
    def getSelectedSize(self):
        return len([p for p in self if p.isSelected()])
    
    def getDiscardedSize(self):
        return len([p for p in self.iterAll() if p.isDiscarded()])
    
    def clear(self):
        self.XIND = 0
        self.YIND = 1
        self.ZIND = 2
        self._points = []


class PathData(Data):
    """ Just contains two list of x and y coordinates. """
    
    def __init__(self, **kwargs):
        Data.__init__(self, **kwargs)
    
    def splitLongestSegment(self):
        """ Split the longest segment by adding the midpoint. """
        maxDist = 0
        n = self.getSize()
        # Find the longest segment and its index
        for i in range(n-1):
            p1 = self.getPoint(i)
            x1, y1 = p1.getX(), p1.getY()
            p2 = self.getPoint(i+1)
            x2, y2 = p2.getX(), p2.getY()
            dist = (x1-x2)**2 + (y1-y2)**2
            if dist > maxDist:
                maxDist = dist
                maxIndex = i+1
                midX = (x1+x2)/2
                midY = (y1+y2)/2
        # Add a midpoint to it
        point = self.createEmptyPoint()
        point.setX(midX)
        point.setY(midY)
        self.addPoint(point, position=maxIndex)
        
    def createEmptyPoint(self):
        data = [0.] * self._dim # create 0, 0...0 point
        point = Point(0, data, 0)
        point._container = self
        
        return point
    
    def removeLastPoint(self):
        del self._points[-1]
