# I can't think of a better way to do this right now...
_simple_labels = {'bird': 0, 'dog': 1, 'reptile': 2}
LABEL_MAPPING = {0: 'dog', 1: 'reptile', 2: 'reptile', 3: 'dog', 4: 'reptile', 5: 'reptile', 6: 'bird',
                 7: 'dog', 8: 'dog', 9: 'dog', 10: 'dog', 11: 'dog', 12: 'reptile', 13: 'bird', 14: 'reptile',
                 15: 'dog', 16: 'reptile', 17: 'dog', 18: 'dog', 19: 'dog', 20: 'dog', 21: 'dog', 22: 'dog',
                 23: 'dog', 24: 'dog', 25: 'dog', 26: 'dog', 27: 'reptile', 28: 'bird', 29: 'reptile', 30: 'reptile',
                 31: 'bird', 32: 'reptile', 33: 'dog', 34: 'dog', 35: 'bird', 36: 'dog', 37: 'bird', 38: 'bird',
                 39: 'reptile', 40: 'bird', 41: 'dog', 42: 'bird', 43: 'bird', 44: 'reptile', 45: 'reptile',
                 46: 'bird', 47: 'reptile', 48: 'reptile', 49: 'dog',50: 'bird', 51: 'bird', 52: 'reptile',
                 53: 'reptile', 54: 'bird', 55: 'reptile', 56: 'bird', 57: 'bird', 58: 'bird', 59: 'bird', 60: 'bird',
                 61: 'reptile', 62: 'reptile', 63: 'bird', 64: 'dog', 65: 'reptile', 66: 'bird', 67: 'bird', 68: 'dog',
                 69: 'bird', 70: 'bird', 71: 'bird', 72: 'bird', 73: 'reptile', 74: 'bird', 75: 'bird', 76: 'dog', 77: 'dog',
                 78: 'reptile', 79: 'dog', 80: 'reptile', 81: 'reptile', 82: 'reptile', 83: 'dog', 84: 'reptile', 85: 'reptile',
                 86: 'bird', 87: 'bird', 88: 'reptile'}

for i in LABEL_MAPPING:
    LABEL_MAPPING[i] = _simple_labels[LABEL_MAPPING[i]]

def convert_labels(subclass_vector: list) -> list:
    '''
    Returns 3-length one-hot list from 89-length one-hot list.
    '''

    list_index = LABEL_MAPPING[subclass_vector.index(1)]
    result = [0] * 3
    result[list_index] = 1

    return result
