def read_plotrc(fname='../reccap2ocean.yaml'):
    import yaml as _yaml

    try:
        rc = _yaml.load(open(fname), Loader=_yaml.SafeLoader)
    except FileNotFoundError:
        print('Could not find the plotting defaults in the standard location')
        print('Please load the data with a manual path defined')
    rc = json2obj(rc)
    return rc


def _json_object_hook(d): 
    from collections import namedtuple
    return namedtuple('plotrc', d.keys())(*d.values())


def json2obj(data): 
    import json
    datastr = str(data).replace("'", '"')
    return json.loads(datastr, object_hook=_json_object_hook)