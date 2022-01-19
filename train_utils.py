import os.path as osp
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from create_dataset_statistics import nvlr_scene_parser, nvlr_translation_parser

COLORS = {
    'black': 0,
    'Black': 0,
    'Yellow': 1,
    'yellow': 1,
    'blue': 2,
    'Blue': 2,
    '#0099ff': 2
}
SHAPES = {
    'square': 0,
    'Square': 0,
    'Block': 0,
    'block': 0,
    'Circle': 1,
    'circle': 1,
    'Triangle': 2,
    'triangle': 2
}
LABEL = {'True': 1,
         'true': 1,
         'false': 0,
         'False': 0}


def _print(something):
    print(something, flush=True)
    return


class BatchSizeScheduler:
    def __init__(self, train_ds, initial_bs, step_size, gamma, max_bs):
        self.train_ds = train_ds
        self.current_bs = initial_bs
        self.max_bs = max_bs
        self.step_size = step_size
        self.gamma = gamma
        self._current_steps = 0

    def reset(self):
        self._current_steps = 0
        return

    def step(self):
        if self.step_size != -1:
            self._current_steps += 1
            if self._current_steps % self.step_size == 0 and self._current_steps > 0:
                self.current_bs = min(self.current_bs * self.gamma, self.max_bs)
            return torch.utils.data.DataLoader(self.train_ds, batch_size=self.current_bs, shuffle=True)
        else:
            return torch.utils.data.DataLoader(self.train_ds, batch_size=self.current_bs, shuffle=True)

    def state_dict(self):
        info = {
            'current_bs': self.current_bs,
            'max_bs': self.max_bs,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'current_steps': self._current_steps
        }
        return info

    def load_state_dict(self, state_dict):
        self.current_bs = state_dict['current_bs']
        self.max_bs = state_dict['max_bs']
        self.step_size = state_dict['step_size']
        self.gamma = state_dict['gamma']
        self._current_steps = state_dict['current_steps']


def analyze_image_slot_of_scene(image_slot):
    xs = []
    ys = []
    shapes = []
    colors = []
    sizes = []

    for item in image_slot:
        x_ = item['x_loc']
        y_ = item['x_loc']
        shape_ = item['type']
        color_ = item['color']
        size_ = item['size']
        xs.append(x_)
        ys.append(y_)
        shapes.append(shape_)
        colors.append(color_)
        sizes.append(size_)

    return xs, ys, shapes, colors, sizes


def nvlr_single_scene_translator(scene: dict, translation: dict):
    n_objects_per_image_slot = [len(scene['structured_rep'][i]) for i in range(3)]
    question_length = len(scene['sentence'].split(' ')) + 2
    question = [1] + [translation[f] for f in scene['sentence'].split(' ')] + [2] + (30 - question_length) * [0]
    question = question[0:30]
    label = LABEL[scene['label']]
    objects_per_scene = []
    for image_slot in range(3):
        n_objects = n_objects_per_image_slot[image_slot]
        objects_per_image_slot = []
        xs, ys, shapes, colors, sizes = analyze_image_slot_of_scene(scene['structured_rep'][image_slot])
        xs_ = [f / 100 for f in xs]
        # scene_xs.append(torch.FloatTensor(xs_ + (7 - n_objects) * [0]).unsqueeze(1))
        ys_ = [f / 100 for f in ys]
        # scene_ys.append(torch.FloatTensor(ys_ + (7 - n_objects) * [0]).unsqueeze(1))
        shapes_ = [SHAPES[f] for f in shapes]
        # scene_shapes.append(torch.LongTensor(shapes_ + (7 - n_objects) * [0]))
        colors_ = [COLORS[f] for f in colors]
        # scene_colors.append(torch.LongTensor(colors_ + (7 - n_objects) * [0]))
        sizes_ = [f / 100 for f in sizes]
        # scene_sizes.append(torch.LongTensor(sizes_ + (7 - n_objects) * [0]))
        for x, y, sh, c, sz in zip(xs_, ys_, shapes_, colors_, sizes_):
            object_ = [x, y, sh, c, sz]
            objects_per_image_slot.append(object_)
        for empty_slots in range((8 - n_objects)):
            objects_per_image_slot.append([-1, -1, -1, -1, -1])
        objects_per_scene.append(objects_per_image_slot)
    # Convert Object Mask and Send Back #
    mask = []
    for f in range(3):
        mask.append([[1] * n_objects_per_image_slot[f] + [0] * (8 - n_objects_per_image_slot[f])])
    # Convert Question Mask and Send Back #
    qmask = [1] * question_length + (30 - question_length) * [0]
    # Pack into a dictionary #
    return objects_per_scene, mask, qmask, question, label


def nvlr_scene_translation(mode='clean_traindev'):
    scenes = nvlr_scene_parser(scenes_path='../clean_data/', mode=mode)
    translation = nvlr_translation_parser()
    dataset_ = {
        'scene_objects': [],
        'mask': [],
        'qmask': [],
        'question': [],
        'label': []
    }
    for s in scenes['scenes'][0]['scenes']:
        objects_per_scene, mask,qmask, question, label = nvlr_single_scene_translator(s, translation)
        # PyTorch Conversion #
        dataset_['scene_objects'].append(torch.FloatTensor(objects_per_scene).unsqueeze(0))
        dataset_['mask'].append(torch.FloatTensor(mask).transpose(1, 0))
        dataset_['qmask'].append(torch.FloatTensor(qmask).unsqueeze(0))
        dataset_['question'].append(torch.LongTensor(question).unsqueeze(0))
        dataset_['label'].append(torch.LongTensor([label]))
    return dataset_


class StateNVLR(Dataset):
    """NVLR dataset made from Scene States."""

    def __init__(self, split='train'):
        if osp.exists(f'../clean_data/{split}_dataset.pt'):
            with open(f'../clean_data/{split}_dataset.pt', 'rb') as fin:
                info = pickle.load(fin)
            self.split = info['split']
            self.x = info['x']
            self.q = info['q']
            self.m = info['m']
            self.qm = info['qm']
            self.y = info['y']
            print("Dataset loaded succesfully!\n")
        else:
            self.split = split
            if self.split != 'test':
                dataset_dict = nvlr_scene_translation(mode='clean_traindev')
            else:
                dataset_dict = nvlr_scene_translation(mode='clean_test')
            self.x = torch.cat(dataset_dict['scene_objects'], dim=0)
            self.q = torch.cat(dataset_dict['question'], dim=0)
            self.m = torch.cat(dataset_dict['mask'], dim=0)
            self.qm = torch.cat(dataset_dict['qmask'], dim=0)
            self.y = dataset_dict['label']
            print("Dataset loaded succesfully!...Saving\n")

            info = {
                'split': self.split,
                'x': self.x,
                'q': self.q,
                'm': self.m,
                'qm': self.qm,
                'y': self.y
            }
            with open(f'../clean_data/{self.split}_dataset.pt', 'wb') as fout:
                pickle.dump(info, fout)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.q[idx], self.m[idx], self.qm[idx], self.y[idx]


def instansiate_train_dev_test(batch_size=4, validation_split=0.2):
    td = StateNVLR(split='traindev')
    te = StateNVLR(split='test')
    train_indices, val_indices = train_test_split(list(range(len(td.y))), test_size=validation_split, stratify=td.y,
                                                  random_state=42)
    train_dataset = torch.utils.data.Subset(td, train_indices)
    val_dataset = torch.utils.data.Subset(td, val_indices)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False)
    return train_data_loader, val_data_loader, test_data_loader
