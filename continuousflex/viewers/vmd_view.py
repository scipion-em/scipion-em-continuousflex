import pyworkflow.utils as pwutils
import pyworkflow.viewer as pwviewer
from continuousflex import Plugin


class VmdView(pwviewer.CommandView):
    """ pwem's VmdView, but VMD_HOME is optional (see Plugin.getVmdEnviron). """
    def __init__(self, vmdCommand, **kwargs):
        pwviewer.CommandView.__init__(self, 'vmd %s' % vmdCommand,
                                      env=Plugin.getVmdEnviron(), **kwargs)

    def show(self):
        pwutils.runJob(None, '', self._cmd, env=Plugin.getVmdEnviron())
