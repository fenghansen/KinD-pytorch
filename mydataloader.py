import os
import gc
import string
import numpy as np
import random
import h5py
import torch
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
from .img_generator import make_shape_parts, sample

class Dataloader():
    def __init__(self, batchsize=8,
                dataname='mini',
                data_dir='./data/mini',
                image_data_format='channels_first', 
                toRAM=True, shuffle=True,
                make_h5_dataset=False, 
                read_h5_dataset=False,):
        
        self.cur_idx = 0
        self.batch_size = batchsize
        self.dataname = dataname
        self.data_dir = data_dir
        self.dir_name = []
        self.image_data_format = image_data_format
        # toRAM是否直接读入内存
        self.toRAM = toRAM
        self.shuffle = shuffle
        # 是否制作或读取数据集
        self.make_h5_dataset = make_h5_dataset
        self.read_h5_dataset = read_h5_dataset
        # 数据存储四老哥
        self.images = []
        self.shapes = []
        self.poses = []
        self.shape_parts = []
        # combinations用于记录组合，e.g. ('00015_1', '00015_2')
        # links用于读入RAM后的记录组合，e.g. (65, 66)
        self.combinations = []
        self.links = []
        # mapping用于映射名字到下标，e.g. {'00015_1': 65}
        self.mapping = {}
        # record用于记录parts，避免重复拆分，加快速度
        self.record = {}
        if not self.read_h5_dataset:
            # 初始化，记录文件结构
            self.data = self.data_struct_init()
            # 制作以图片为单位的path集，标签为图片id，e.g. {'00015_1': path}
            self.path_pair_asPicture = self.make_path_pair_asPicture()
            self.num_of_pairs = len(self.path_pair_asPicture)
            # 数据清洗，去除不匹配数据
            self.data_clear()
            self.filenames = [key for key in self.path_pair_asPicture]
            # 制作以人为单位的path集，标签为人id
            # e.g. {'00015'：{'00015_1':path}, {'00015_2':path}}
            self.path_pair_asPerson = self.make_path_pair_asPerson()
        else:
            self.f = h5py.File(os.path.join(self.data_dir, self.dataname+'.h5'),'r')
            self.read_dataset()
            self.num_of_pairs = len(self.filenames)

        self.idx_order = np.array(range(len(self.links)//batchsize))
        if self.shuffle:
            np.random.shuffle(self.idx_order)
        if self.toRAM and not self.read_h5_dataset:
            self.From_path_to_RAM()
        if self.make_h5_dataset and self.toRAM:
            self.make_dataset()
        self.lenth = len(self.links)

    def __len__(self):
        return self.lenth

    def __getitem__(self, id):
        """数组式调用"""
        if self.toRAM:
            # self.links映射id到下标, 所有数据都在内存里
            index_x, index_y = self.links[id]
            images_x, shapes_x, poses_x = self.images[index_x], self.shapes[index_x], self.poses[index_x]
            parts_x = self.get_shape_parts(index_x, shapes_x, mode='x')
            images_y, shapes_y, poses_y = self.images[index_y], self.shapes[index_y], self.poses[index_y]
            parts_y = self.get_shape_parts(index_y, shapes_y, mode='y')
        elif self.read_h5_dataset:
            # 直接从数据集.h5中读取数据
            index_x, index_y = self.links[id]
            # data_x, data_y = self.data_from_dataset(index_x, index_y)
            images_x = np.array(self.f['images'][index_x])
            shapes_x = np.array(self.f['shapes'][index_x])
            poses_x = np.array(self.f['poses'][index_x])
            parts_x = self.get_shape_parts(index_x, shapes_x, mode='x')
            images_y = np.array(self.f['images'][index_y])
            shapes_y = np.array(self.f['shapes'][index_y])
            poses_y = np.array(self.f['poses'][index_y])
            parts_y = self.get_shape_parts(index_y, shapes_y, mode='y')
        else:
            # 从文件读取图片数据
            # 返回值为两个pair的id('str')
            id_x, id_y = self.combinations[id]
            images_x, poses_x, shapes_x = self.load_data(self.path_pair_asPicture[id_x])
            images_y, poses_y, shapes_y = self.load_data(self.path_pair_asPicture[id_y])
            parts_x = self.get_shape_parts(id_x, shapes_x, mode='x')
            parts_y = self.get_shape_parts(id_y, shapes_y, mode='y')

        # 返回值为两个image_pair
        return images_x, poses_x, shapes_x, parts_x, images_y, poses_y, shapes_y, parts_y
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """迭代器，下一个"""
        # 超限重启，shuffle模式下会重新生成一个排列
        index = self.idx_order[self.cur_idx] * self.batch_size
        if self.cur_idx + 1 >= len(self.idx_order):
            self.cur_idx = 0
            if self.shuffle:
                np.random.shuffle(self.idx_order)
            index = self.idx_order[self.cur_idx] * self.batch_size
        images_x = []; shapes_x = []; poses_x = []; parts_x = []
        images_y = []; shapes_y = []; poses_y = []; parts_y = []
        for i in range(self.batch_size):
            image_x, pose_x, shape_x, part_x, image_y, pose_y, \
                shape_y, part_y = self.__getitem__(index + i)
            images_x.append(image_x)
            shapes_x.append(shape_x)
            poses_x.append(pose_x)
            parts_x.append(part_x)
            images_y.append(image_y)
            shapes_y.append(shape_y)
            poses_y.append(pose_y)
            parts_y.append(part_y)
        images_x = np.array(images_x)
        shapes_x = np.array(shapes_x)
        poses_x = np.array(poses_x)
        parts_x = np.array(parts_x)
        images_y = np.array(images_y)
        shapes_y = np.array(shapes_y)
        poses_y = np.array(poses_y)
        parts_y = np.array(parts_y)
        self.cur_idx += 1
        return images_x, poses_x, shapes_x, parts_x, images_y, poses_y, shapes_y, parts_y
            
    def data_struct_init(self,):
        data = {}
        for root, dirs, files in os.walk(self.data_dir):
            for name in dirs:   # 记录一下文件夹名(的顺序)
                self.dir_name.append(name)
            if len(dirs) == 0:  # 都是图片的目录
                name = root[len(self.data_dir)+1 : ]
                data[name] = files
        print(">> dir_name list: {}".format(self.dir_name))
        return data

    def make_path_pair_asPicture(self,):
        # 以真实图片为单位制作path_pair
        pairs = {}
        for root, dirs, files in os.walk(self.data_dir):
            if len(dirs) > 0: continue # 不是纯图片目录就跳
            for name in files:
                id = name[:7]
                path = os.path.join(root, name)
                if id not in pairs:
                    pairs[id] = []
                pairs[id].append(path)
        print(">> {} picture pair with path has been found...".format(len(pairs)))
        return pairs
    
    def make_path_pair_asPerson(self,):
        """以人为单位制作path_pair"""
        pairs = {}
        idx = 0
        self.combinations = []
        for key in self.path_pair_asPicture:
            # 将人的编号作为id，划分出这个人的pair list
            id = key[:5]
            if id not in pairs:
                pairs[id] = {}
            # num = key[6]    # 方向编号，类型为str
            pairs[id][key] = self.path_pair_asPicture[key]
            self.mapping[key] = idx
            idx += 1
        
        for key in pairs:
            # num_of_pose = len(pairs[key])
            # 记录自己对自己的全排列映射顺序
            for pose_i in pairs[key]:
                for pose_j in pairs[key]:
                    self.combinations.append((pose_i, pose_j))
        
        # 直接构建位置映射，加快速度
        for x, y in self.combinations:
            self.links.append((self.mapping[x], self.mapping[y]))

        print(">> There are {} kinds of combines for training".format(len(self.combinations)))
        print(">> {} legal Person-ID with path has been found...".format(len(pairs)))
        return pairs
    
    def data_clear(self, can_del=False):
        # 数据清洗，不配对的直接删 
        maxLenth = 0
        has_changed = False
        for key in self.path_pair_asPicture:
            lenth = len(self.path_pair_asPicture[key])
            maxLenth = max(lenth, maxLenth)
        
        for key in self.path_pair_asPicture:
            if len(self.path_pair_asPicture[key]) < maxLenth:
                has_changed = True
                for path in self.path_pair_asPicture[key]:
                    print("目标数据不匹配：",path)
                    if can_del:
                        os.remove(path)
                    else:
                        raise OSError
                if can_del: del self.path_pair_asPicture[key]
        
        if has_changed:
            self.data = self.data_struct_init()
    
    def From_path_to_RAM(self,):
        """将所有pair读入内存"""
        images = []
        shapes = []
        poses = []
        idx = 0
        print(">> Loading data from path to RAM...")
        with tqdm(total=self.num_of_pairs) as pbar:
            for key in self.path_pair_asPicture:
                # 读入数据对,并记录从key（图片id）到数组下标的映射
                # 测试时暂无pose和shape_parts
                image, pose, shape = self.load_data(self.path_pair_asPicture[key])
                images.append(image)
                shapes.append(shape)
                poses.append(pose)  
                idx += 1
                pbar.update(1)
        
        self.images = np.array(images)
        self.shapes = np.array(shapes)
        self.poses  = np.array(poses)
    
    def get_shape_parts(self, index, shapes, mode='x'):
        # 加快读取，便捷操作
        if index not in self.record:
            # 3300对应大概1G内存
            if (self.shuffle and len(self.record)>8000
                or self.shuffle is False and mode=='x'):
                del self.shape_parts
                self.shape_parts = []
                self.record.clear()
                self.record[index] = 0
                parts = make_shape_parts(shapes) 
                self.shape_parts.append(parts)
            else: # mode = 'y', 这个是新的
                self.record[index] = len(self.record)
                parts = make_shape_parts(shapes) 
                self.shape_parts.append(parts)
        else:
            parts = self.shape_parts[self.record[index]]

        return parts

    def name2num(self, filenames):
        return np.array([int(name[:5])*10 + int(name[-1]) for name in filenames])
    def num2name(self, nums):
        return [str.format("%05d"%(num // 10)) + '_' 
                + str.format("%05d"%(num % 10))  for num in nums]

    def make_dataset(self,):
        dir = self.data_dir; name = self.dataname
        # 创建一个h5文件，文件指针是f
        with h5py.File(os.path.join(dir, name+'.h5'),'w') as f:
            num = self.name2num(self.filenames)
            f.create_dataset('filenames', data=num)
            f.create_dataset('links',  data=self.links)
            print("--> Saving Images...")
            f.create_dataset('images', data=self.images, chunks=(1, 3, 256, 256))
            print("--> Saving Shapes...")
            f.create_dataset('shapes', data=self.shapes, chunks=(1, 3, 256, 256))
            print("--> Saving Poses...")
            f.create_dataset('poses',  data=self.poses, chunks=(1, 3, 256, 256))
            print(">> Successfully Create H5 Dataset!!")
    
    def read_dataset(self,):
        print(">> Loading h5 dataset!!")
        self.filenames = np.array(self.num2name(self.f['filenames']))
        self.links  = np.array(self.f['links'])
        if self.toRAM:  # 内存16G以上，图片可以直接读入内存
            self.images = np.array(self.f['images']).astype(np.float16)
            print("--> Images Compelete")
            self.shapes = np.array(self.f['shapes']).astype(np.float16)
            print("--> Shapes Compelete")
            self.poses = np.array(self.f['poses']).astype(np.float16)
            print("--> Poses Compelete")
            print(">> Successfully Read All H5 Dataset!!")
        else:
            print(">> Gradually Read Data From H5 Dataset!!")
    
    def data_from_dataset(self, idx, idy):
        # print(">> {} data generator has built...".format(name))
        images_x = np.array(self.f['images'][idx])
        shapes_x = np.array(self.f['shapes'][idx])
        images_y = np.array(self.f['images'][idy])
        shapes_y = np.array(self.f['shapes'][idy])
        # images_x = self.f['images'][idx].astype('uint8')
        # shapes_x = self.f['shapes'][idx].astype('uint8')
        # images_y = self.f['images'][idy].astype('uint8')
        # shapes_y = self.f['shapes'][idy].astype('uint8')
        return (images_x, shapes_x), (images_y, shapes_y)

    def load_data(self, path_list):
        """读取一个pair的数据，顺序为image, pose, shape, shape_parts"""
        pair_imgs = []
        for path in path_list:
            img = Image.open(path)
            img = np.array(img).astype(np.float16) / 255.
            if self.image_data_format == 'channels_first':
                img = img.transpose(2, 0, 1)
            pair_imgs.append(img)
        return np.array(pair_imgs)                

    @staticmethod
    def scale_imgs(imgs):
        """Scale images from [0,255] to [0,1]"""
        return imgs / 255.

    @staticmethod
    def unscale_imgs(imgs):
        """Un-scale images from [0,1] to [0,255]"""
        return imgs * 255

    # @staticmethod
    # def scale_tanh_imgs(imgs):
    #     """Scale images from [0,255] to [-1,1]"""
    #     return imgs / 127.5 - 1

    # @staticmethod
    # def unscale_tanh_imgs(imgs):
    #     """Un-scale images from [-1,1] to [0,255]"""
    #     return (imgs + 1.) * 127.5

class Stage2Dataloader(Dataloader):
    def __next__(self):
        images_x, poses_x, shapes_x, parts_x, images_y, \
            poses_y, shapes_y, parts_y = super().__next__()
        images_x = torch.from_numpy(images_x)
        shapes_x = torch.from_numpy(shapes_x)
        poses_x = torch.from_numpy(poses_x)
        parts_x = torch.from_numpy(parts_x)
        images_y = torch.from_numpy(images_y)
        shapes_y = torch.from_numpy(shapes_y)
        poses_y = torch.from_numpy(poses_y)
        parts_y = torch.from_numpy(parts_y)
        return (images_x, poses_x, shapes_x, parts_x), (images_y, poses_y, shapes_y, parts_y)

if __name__ == "__main__":
    count = 0
    total = 64000
    # loader = Dataloader(data_dir='./data/test',toRAM=True, shuffle=True,
                        # make_h5_dataset=True, read_h5_dataset=False)
    loader = Stage2Dataloader(data_dir='./data/mini', toRAM=False, shuffle=False,
                        make_h5_dataset=False, read_h5_dataset=True,
                        dataname="mini", batchsize=1)
    with tqdm(total=total) as pbar:
        for data_x, data_y in loader:
            image, pose, shape, parts = data_y
            count += loader.batch_size
            pbar.update(loader.batch_size)
            sample(image[-1], figure_size=(1, 3), num=0)
            sample(pose[-1], figure_size=(1, 3), num=1)
            sample(shape[-1], figure_size=(1, 3), num=2)
            if count % 10 == 0:
                # 此处用于设置断点debug查看
                count = count
            if count >= total: break

    # test = Dataloader('./data/test')