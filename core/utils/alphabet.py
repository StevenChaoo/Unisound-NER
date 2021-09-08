"""
Alphatet maps objects to integer ids. It provides tow way mapping from the index to the objects.
"""
import json
import os


class Alphabet:
    def __init__(self, name, label=False, keep_growing=True):
        self.__name = name
        self.UNKNOWN = '</unk>'
        self.label = label
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = -1
        self.next_index = 0
        if not self.label:
            self.add("[PAD]")
            self.add(self.UNKNOWN)
            self.add("[CLS]")
            self.add("[SEP]")

    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        
        # Index 0 is occupied by default, all else following.
        self.default_index = -1
        self.next_index = 0

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        try:
            return self.instances[index]
        except IndexError:
            print('WARNING: Alphabet get_instance, unknown instance, return the first label.')
            raise Exception
        
    def convert_tokens_to_ids(self, tokens):
        return [self.get_index(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.get_instance(_id) for _id in ids]
    
    def size(self):
        return len(self.instances) + 1

    def iteritems(self):
        for item in self.instance2index.items():
            yield item

    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError('Enumerate is allowed between [1: size of the alphabet)')
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        self.keep_growing = False
    
    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data['instances']
        self.instance2index = data['instance2index']

    def save(self, output_directory, name=None):
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name +  '.json'), 'w'))
        except Exception as e:
            print('Exception: Alphabet is not saved: ' % repr(e))

    def load(self, input_directory, name=None):
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
        self.next_index = len(self.instances)


if __name__ == '__main__':
    alphabet = Alphabet(name="char_alphabet")
    with open('../data/embeddings/gigaword_chn.all.a2b.uni.ite50.vec') as f:
        for line in f:
            word = line.rstrip('\n').split(' ')[0]
            alphabet.add(word)
    with open('../data/bingli/train.data') as f:
        for line in f:
            word = line.rstrip('\n').split(' ')[0]
            alphabet.add(word)
    with open('../data/cdd-check/top1k/train.txt') as f:
        for line in f:
            word = line.rstrip('\n').split(' ')[0]
            alphabet.add(word)
    with open('../data/cdd-check/top1k_2k/data.txt') as f:
        for line in f:
            word = line.rstrip('\n')
            alphabet.add(word)
    alphabet.save(output_directory="../data/embeddings")
