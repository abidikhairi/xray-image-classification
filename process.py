import os
import pandas as pd


if __name__ == '__main__':
	basedir = os.path.realpath('./data')
	splits = ['train', 'test', 'val']
	categories = ('NORMAL', 'PNEUMONIA')

	for split in splits:
		data = {
			'image': [],
			'label': []
		}
		for label, category in enumerate(categories):
			_dir = os.path.join(basedir, split, category)
			for filename in os.listdir(_dir):
				path = os.path.join(_dir, filename)

				data['image'].append(path)
				data['label'].append(label)

		pd.DataFrame(data).to_csv(os.path.join(basedir, f'{split}.csv'), index=False)
