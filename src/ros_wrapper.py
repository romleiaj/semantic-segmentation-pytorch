#! /usr/bin/python3

# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# ROS imports
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode, find_recursive, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from config import cfg

cwd = os.path.dirname(os.path.realpath(__file__))
colors = loadmat('%s/data/color150.mat' % cwd)['colors']
names = {}
with open('%s/data/object150_info.csv' % cwd) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

class SegmentImage():
    def __init__(self, cfg, gpu, img_in, img_out):
        self.cfg = cfg
        self.gpu = gpu
        self.img_in = img_in
        self.img_out = img_out
        self.bridge = CvBridge() 

    def imresize(self, im, size, interp='bilinear'):
        if interp == 'nearest':
            resample = Image.NEAREST
        elif interp == 'bilinear':
            resample = Image.BILINEAR
        elif interp == 'bicubic':
            resample = Image.BICUBIC
        else:
            raise Exception('resample method undefined!')

        return im.resize(size, resample)


    def visualize_result(self, data, pred, cfg):
        (img, info) = data
    
        # print predictions in descending order
        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        print("Predictions in [{}]:".format(info))
        for idx in np.argsort(counts)[::-1]:
            name = names[uniques[idx] + 1]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print("  {}: {:.2f}%".format(name, ratio))
    
        # colorize prediction
        pred_color = colorEncode(pred, colors).astype(np.uint8)
    
        # aggregate images and save
        im_vis = np.concatenate((img, pred_color), axis=1)
    
        img_name = info.split('/')[-1]
        Image.fromarray(im_vis).save(
            os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))

        # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    
    def test(self, loader):
        self.segmentation_module.eval()
    
        pbar = tqdm(total=len(loader))
        for batch_data in loader:
            # process data
            batch_data = batch_data[0]
            segSize = (batch_data['img_ori'].shape[0],
                       batch_data['img_ori'].shape[1])
            img_resized_list = batch_data['img_data']
    
            with torch.no_grad():
                scores = torch.zeros(1, self.cfg.DATASET.num_class, segSize[0], segSize[1])
                scores = async_copy_to(scores, self.gpu)
    
                for img in img_resized_list:
                    feed_dict = batch_data.copy()
                    feed_dict['img_data'] = img
                    del feed_dict['img_ori']
                    del feed_dict['info']
                    feed_dict = async_copy_to(feed_dict, self.gpu)
    
                    # forward pass
                    pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                    scores = scores + pred_tmp / len(self.cfg.DATASET.imgSizes)
    
                _, pred = torch.max(scores, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())
    
            # visualization
            visualize_result(
                (batch_data['img_ori'], batch_data['info']),
                pred,
                self.cfg
            )
    
            pbar.update(1)
    
    def image_callback(self, img):
        tic = rospy.Time.now()
        rospy.loginfo("Processing image...")
    
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding="rgb8")
        except CvBridgeError as e:
            rospy.logerr(e)

        # Need it in PIL?
        PILimg = Image.fromarray(cv_image)
        ori_width, ori_height = cv_image.size
    
        img_resized_list = []
        for this_short_size in self.cfg.DATASET.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.cfg.DATASET.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)
        
            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)
    
            # resize images
            img_resized = self.imresize(PILimg, (target_width, target_height), interp='bilinear')
        
            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)
    
    
        dataset = dict()
        dataset['img_ori'] = np.asarray(PILimg)
        dataset['img_data'] = [x.contiguous() for x in img_resized_list]
        dataset['info'] = img.header
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.TEST.batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=5,
            drop_last=True)

       # img_labels = self.segment(gpu)
    
        # Main loop
        test(load)
    
        rospy.loginfo('Inference done in %s seconds' % (rospy.Time.now() - tic))
    
    def main(self):
    
        torch.cuda.set_device(self.gpu)
    
        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch=self.cfg.MODEL.arch_encoder,
            fc_dim=self.cfg.MODEL.fc_dim,
            weights=self.cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(
            arch=self.cfg.MODEL.arch_decoder,
            fc_dim=self.cfg.MODEL.fc_dim,
            num_class=self.cfg.DATASET.num_class,
            weights=self.cfg.MODEL.weights_decoder,
            use_softmax=True)
    
        crit = nn.NLLLoss(ignore_index=-1)
    
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        self.segmentation_module.cuda()

        self.seg_pub = rospy.Publisher(self.img_out, sensor_msgs.msg.Image, queue_size=10)
        rospy.Subscriber(self.img_in, sensor_msgs.msg.Image, self.image_callback)
    
        rospy.loginfo("Listening for image messages on topic %s..." % self.img_in)
        rospy.loginfo("Publishing segmented images to topic %s..." % self.img_out)
        rospy.spin()


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'
        
    rospy.init_node("segmentation")

    yamlcfg = rospy.get_param('~cfg')
    gpu = rospy.get_param('~gpu', 0)
    img_in = rospy.get_param('~img_in', "image_raw")
    img_out = rospy.get_param('~img_out', "image_segmented")

    cfg.merge_from_file(yamlcfg)
    # cfg.freeze()

    rospy.loginfo("Loaded configuration file {}".format(yamlcfg))
    rospy.loginfo("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    print(cfg)

    cfg.DIR = ("%s/%s" % 
                    (os.path.dirname(os.path.dirname(yamlcfg)),
                    cfg.DIR.split('/')[1]))
    print(cfg.DIR)

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)
    print(cfg.MODEL.weights_encoder)


    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exist!"
    
    SI = SegmentImage(cfg, gpu, img_in, img_out)
    SI.main()
