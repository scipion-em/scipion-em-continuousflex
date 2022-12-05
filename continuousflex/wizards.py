from pwem.constants import *
from pwem.wizards import *
from continuousflex.protocols.protocol_denoise_volumes import FlexProtVolumeDenoise

class FlexFilterVolumesWizard(FilterVolumesWizard):
    _targets = [(FlexProtVolumeDenoise, ['lowFreqDig','freqDecayDig'])]

    def _getParameters(self, protocol):
        protParams = {}

        labels = ['lowFreqDig', 'highFreqDig', 'freqDecayDig']
        protParams['unit'] = UNIT_PIXEL

        values = [protocol.getAttributeValue(l) for l in labels]

        protParams['input']= protocol.inputVolumes
        protParams['label']= labels
        protParams['value']= values
        protParams['mode'] = 0 # FILTER_SPACE_FOURIER = 0
        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return FilterVolumesWizard._getListProvider(self, _objs)

    def show(self, form):
        params = self._getParameters(form.protocol)
        _value = params['value']
        _label = params['label']
        _mode = params['mode']
        _unit = params['unit']
        FilterVolumesWizard.show(self, form, _value, _label, _mode, _unit)