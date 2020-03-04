import gzip
import csv
import numpy as np

class DataLoader:
    def __init__(self):
        data_path = '../data/letter.data.gz'
        lines = self._read(data_path)
        data, target = self._parse(lines)
        self.data, self.target = self._pad(data, target)

    @staticmethod
    def _read(filepath):
        with gzip.open(filepath, 'rt') as file_:
            reader = csv.reader(file_, delimiter='\t')
            lines = list(reader)
            return lines

    @staticmethod
    def _parse(lines):
        lines = sorted(lines, key=lambda x: int(x[0]))
        data, target = [], []
        next_ = None

        for line in lines:
            if not next_:
                data.append([])
                target.append([])
            else:
                assert next_ == int(line[0])
            next_ = int(line[2]) if int(line[2]) > -1 else None
            pixels = np.array([int(x) for x in line[6:134]])
            pixels = pixels.reshape((16, 8))
            data[-1].append(pixels)
            target[-1].append(line[1])
        return data, target

    @staticmethod
    def _pad(data, target):
        """
        Add padding to ensure word length is consistent
        """
        max_length = max(len(x) for x in target)
        padding = np.zeros((16, 8))
        data = [x + ([padding] * (max_length - len(x))) for x in data]
        target = [x + ([''] * (max_length - len(x))) for x in target]
        return np.array(data), np.array(target)

def get_dataset():
    dataset = DataLoader()

    # Flatten images into vectors.
    dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))

     # One-hot encode targets.
    target = np.zeros(dataset.target.shape + (26,))
    for index, letter in np.ndenumerate(dataset.target):
        if letter:
            target[index][ord(letter) - ord('a')] = 1
    dataset.target = target

    # Shuffle order of examples.
    order = np.random.permutation(len(dataset.data))
    dataset.data = dataset.data[order]
    dataset.target = dataset.target[order]
    return dataset
