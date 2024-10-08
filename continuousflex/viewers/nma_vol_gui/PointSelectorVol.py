# **************************************************************************
# *
# * Authors:    Mohamad Harastani            (mohamad.harastani@igbmc.fr)
# *             Slavica Jonic                (slavica.jonic@upmc.fr)
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

from continuousflex.viewers.plotter_vol import plotArray2D_xy
from math import sqrt


class PointSelectorVol():
    """ Graphical manager based on Matplotlib to handle mouse
    events of click, drag and release and mark some point
    from input Data as 'selected'.
    """

    def __init__(self, ax, data, callback=None, LimitL=None, LimitH=None, alpha=None, s=None):
        self.ax = ax
        self.data = data
        self.LimitL = LimitL
        self.LimitH = LimitH
        # Point size and transparancy:
        self.alpha = alpha.get()
        self.s = s.get()
        self.createPlots(ax)
        self.press = None
        self.callback = callback

        # connect to all the events we need
        self.cidpress = self.rectangle_selection.figure.canvas.mpl_connect(
            'button_press_event', self.onPress)
        self.cidrelease = self.rectangle_selection.figure.canvas.mpl_connect(
            'button_release_event', self.onRelease)
        self.cidmotion = self.rectangle_selection.figure.canvas.mpl_connect(
            'motion_notify_event', self.onMotion)

    def createPlots(self, ax):
        # plotArray2D(ax, self.data)
        # To avoid plotting the sidebar twice
        if(self.LimitL):
            plotArray2D_xy(ax, self.data, vvmin=self.LimitL, vvmax=self.LimitH, alpha=self.alpha, s=self.s)
        else:
            plotArray2D_xy(ax, self.data, alpha=self.alpha, s=self.s)
        self.createSelectionPlot(ax)

    def getSelectedData(self):
        xs, ys = [], []
        for point in self.data:
            if point.getState() == 1:  # point selected
                xs.append(point.getX())
                ys.append(point.getY())
        return xs, ys

    def createSelectionPlot(self, ax):
        xs, ys = self.getSelectedData()
        markersize = None
        if self.s:
            markersize = sqrt(self.s)
        self.plot_selected, = ax.plot(xs, ys, 'o', ms=markersize, alpha=0.4,
                                      color='yellow')

        self.rectangle_selection, = ax.plot([0], [0], ':')  # empty line

    def onPress(self, event):
        if event.inaxes != self.rectangle_selection.axes:
            return
        # ignore click event if toolbar is active
        if self.ax.get_navigate_mode():
            return

        self.press = True
        self.originX = event.xdata
        self.originY = event.ydata

    def onMotion(self, event):
        if self.press is None: return
        self.update(event, addSelected=False)

    def onRelease(self, event):
        self.press = None
        self.update(event, addSelected=True)

    def inside(self, x, y, xmin, xmax, ymin, ymax):
        return (xmin <= x <= xmax and
                ymin <= y <= ymax)

    def update(self, event, addSelected=False):
        """ Update the plots with selected points.
        Take the selection rectangle and include points from 'event'.
        If addSelected is True, update the data selection.
        """
        # ignore click event if toolbar is active
        if self.ax.get_navigate_mode():
            return
        ox, oy = self.originX, self.originY
        ex, ey = event.xdata, event.ydata
        xs1, ys1 = self.getSelectedData()

        x1 = min(ox, ex)
        x2 = max(ox, ex)
        y1 = min(oy, ey)
        y2 = max(oy, ey)

        if addSelected:
            xs, ys = [0], [0]  # rectangle selection
        else:
            xs = [x1, x2, x2, x1, x1]
            ys = [y1, y1, y2, y2, y1]

        for point in self.data:
            x, y = point.getX(), point.getY()
            if self.inside(x, y, x1, x2, y1, y2):
                xs1.append(x)
                ys1.append(y)
                if addSelected:
                    point.setSelected()

        if self.callback:  # Notify changes on selection
            self.callback()
        self.plot_selected.set_data(xs1, ys1)
        self.rectangle_selection.set_data(xs, ys)
        self.ax.figure.canvas.draw()
