#! /usr/bin/python3

# System libs
import os
import argparse
from distutils.version import LooseVersion
import queue
import threading
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# ROS imports
import rospy
# Have to import this way to prevent name conflicts with PIL
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
# Our libs
from dataset import MemDataset
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
        self.time_ori = rospy.Duration(0)
        self.frame_id = ""
        self.bridge = CvBridge() 
        self.loader_q = queue.Queue(1)
        self.ready = True
        self.PERSON = 12 # index of persons
        # [sidewalk, path, runway, road, floor]
        self.DRIVEABLE = [11, 52, 54, 6, 3]


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
    
        # aggregate images and publish
        im_vis = np.concatenate((img, pred_color), axis=1)
        
        img_msg = self.bridge.cv2_to_imgmsg(im_vis, encoding='rgb8')
        self.seg_pub.publish(img_msg)
    
        #img_name = "test.jpg"
        #Image.fromarray(im_vis).save(
        #    os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))

    def run_inference(self, loader):
        rospy.loginfo("Processing image...")
        tic = rospy.get_rostime()

        self.segmentation_module.eval()
    
        pbar = tqdm(total=len(loader))
        # process data
        for batch_data in loader:
            batch_data = batch_data[0]
            h, w = batch_data['img_ori'].shape[:2]
            segSize = (h, w)
            new_img = np.zeros((h, w, 3))
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
                    pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
                    scores = scores + pred_tmp / len(self.cfg.DATASET.imgSizes)
    
                #_, pred = torch.max(scores, dim=1)
                #pred = as_numpy(pred.squeeze(0).cpu())
                nparr = as_numpy(scores.squeeze(0).cpu())
                
            # Putting drivable in green channel
            new_img[:, :, 1] = np.sum(nparr[self.DRIVEABLE], axis=0)
            # Person in red channel
            new_img[:, :, 0] = nparr[self.PERSON, :, :]
            # Converting to uint8
            uint_img = (new_img * 255).astype('uint8')
            # Placing original and segmented image side-by-side
            im_vis = np.concatenate((batch_data['img_ori'], uint_img), axis=1)
            img_msg = self.bridge.cv2_to_imgmsg(im_vis, encoding='rgb8')
            img_msg.header.frame_id = self.frame_id
            img_msg.header.stamp = self.time_ori
            self.seg_pub.publish(img_msg)
    
            # visualization
            #self.visualize_result(
            #    (batch_data['img_ori'], batch_data['info']),
            #    pred2,
            #    self.cfg
            #)
            pbar.update(1)
        
        rospy.loginfo('Image latency of %.03f seconds.' % 
                ((rospy.get_rostime() - self.time_ori).to_sec()))
    
    def image_callback(self, img):
        # Only want to update if there isn't an image already being processed
        if not self.ready:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding="rgb8")
        except CvBridgeError as e:
            rospy.logerr(e)

        imgs = []
        # Need it in PIL?
        PILimg = Image.fromarray(cv_image)
        PILimg = PILimg.resize((1000, 600))
        # In case we ever want to batch multiple images
        imgs.append(PILimg)
        img_list = [{'img': x} for x in imgs]

        dataset = MemDataset(
                img_list,
                cfg.DATASET
                )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.TEST.batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=5,
            drop_last=True)

       # img_labels = self.segment(gpu)
        self.time_ori = img.header.stamp
        self.frame_id = img.header.frame_id
        self.loader_q.put(loader)
    
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

        self.seg_pub = rospy.Publisher(self.img_out, sensor_msgs.msg.Image, queue_size=1)
        rospy.Subscriber(self.img_in, sensor_msgs.msg.Image, self.image_callback)
    
        rospy.loginfo("Listening for image messages on topic %s..." % self.img_in)
        rospy.loginfo("Publishing segmented images to topic %s..." % self.img_out)

        rospy.loginfo("Waiting for loader from queue...")
        while not rospy.is_shutdown():
            rospy.sleep(0.01)
            try:
                loader = self.loader_q.get_nowait()
                self.ready = False
                self.run_inference(loader)
                self.ready = True
            except queue.Empty:
                pass

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

    cfg.DIR = ("%s/%s" % 
                    (os.path.dirname(os.path.dirname(yamlcfg)),
                    cfg.DIR.split('/')[1]))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)


    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exist!"
    
    SI = SegmentImage(cfg, gpu, img_in, img_out)
    SI.main()
