from . import maps

import yaml as _yaml

_config = _yaml.load(open('../../reccap2ocean.yaml'), Loader=_yaml.SafeLoader)
colors = _config['colors']
figw1 = _config['figwidth']['single']
figw2 = _config['figwidth']['double']
