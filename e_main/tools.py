import json
import os
import pathlib

from e_main import const
from e_main.preset import Preset


def write_jsonx(data, fpath, mode='w'):
    suffix = pathlib.Path(fpath).suffix

    if suffix == '.jsonl':
        with open(fpath, mode, encoding=const.ecd_utf8) as file:
            for item in data:
                json_item = json.dumps(item, ensure_ascii=False)
                file.write(json_item + '\n')
    elif suffix == '.json':
        with open(fpath, mode, encoding=const.ecd_utf8) as file:
            json.dump(data, file, ensure_ascii=False, indent=1)


def read_jsonx(fpath):
    suffix = pathlib.Path(fpath).suffix

    if suffix == '.jsonl':
        data = []
        with open(fpath, 'r', encoding=const.ecd_utf8) as f:
            for i, line in enumerate(f.readlines()):
                data.append(json.loads(line))
        return data

    elif suffix == '.json':
        with open(fpath, 'r', encoding=const.ecd_utf8) as file:
            data = json.load(file)
        return data


if __name__ == '__main__':
    pass

    fpath_jsonl = os.path.join(Preset.root, 'sample', 'test.jsonl')
    fpath_json = os.path.join(Preset.root, 'sample', 'test.json')
    data = [{'a': 1, 'b': 2}, {'c': 3, 'd': 4}]
    write_jsonx(data, fpath_jsonl)
    write_jsonx(data, fpath_json)
    data_jsonl = read_jsonx(fpath_jsonl)
    data_json = read_jsonx(fpath_json)
