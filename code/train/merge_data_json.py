import json

ann1 = json.load(open('../../tcdata/tile_round2_train/train_annos.json', 'r'))
ann2 = json.load(open('../../tcdata/tile_round2_train/train_annos_20210208.json', 'r'))
ann_to_save = '../../tcdata/tile_round2_train/merged_train_annos.json'

ann1.extend(ann2)
with open(ann_to_save, 'w') as fp:
    json.dump(ann1, fp, indent=4, separators=(',', ': '))