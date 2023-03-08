import json
import os
import pathlib
import pandas as pd
from datetime import timedelta, datetime

from e_main import const
from e_main.preset import Preset


def save_jsonx(data, fpath, mode='w'):
    suffix = pathlib.Path(fpath).suffix

    if suffix == '.jsonl':
        with open(fpath, mode, encoding=const.ecd_utf8) as file:
            for item in data:
                json_item = json.dumps(item, ensure_ascii=False)
                file.write(json_item + '\n')
                file.flush()
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


class Watch:
    def __init__(self, fctn=pd.Timestamp.now):
        self.fctn = fctn
        self.tick = self.fctn()
        self.records = []

    def see(self):
        tick = self.fctn()
        res = tick - self.tick
        self.records.append(res)
        self.tick = tick
        return res

    @staticmethod
    def pdtd2HMS(pdtd):
        ts = pdtd.total_seconds()
        hrs, ts = divmod(ts, 3600)
        mins, ts = divmod(ts, 60)
        secs = round(ts)

        hrs, mins, secs = [str(int(n)).zfill(2) for n in [hrs, mins, secs]]
        tds = f'{hrs}:{mins}:{secs}'
        return tds

    def seeSec(self):
        return round(self.see().total_seconds(), 6)

    def total(self):
        totalTime = sum(self.records, timedelta())
        return totalTime

    @staticmethod
    def now():
        t = datetime.now()
        rt = datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
        return pd.Timestamp(rt)


def watch_time(func):
    def wrap(*args, **kwargs):
        w = Watch()
        res = func(*args, **kwargs)

        print(f"Time Cost : {func.__name__} {w.see()}")
        return res

    return wrap



if __name__ == '__main__':
    pass

    # sample of read/write json/jsonl
    fpath_jsonl = os.path.join(Preset.root, 'sample', 'test.jsonl')
    fpath_json = os.path.join(Preset.root, 'sample', 'test.json')
    data = [{'a': 1, 'b': 2}, {'c': 3, 'd': 4}]
    save_jsonx(data, fpath_jsonl)
    save_jsonx(data, fpath_json)
    data_jsonl = read_jsonx(fpath_jsonl)
    data_json = read_jsonx(fpath_json)
