from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy.misc import imresize
import os
import cv2
from lib.model.config import cfg
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import flags
FLAGS = flags.FLAGS

class AttackGraph(object):
    def __init__(self):
        # shape setup
        image_shape = (FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels)
        input_shape = (FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels)

        # normalization for faster-rcnn
        self.pixel_low = (0.0 - cfg.PIXEL_MEANS) / 255.0
        self.pixel_high = (255.0 - cfg.PIXEL_MEANS) / 255.0

        # mask files
        mask_files = [x.strip() for x in FLAGS.attack_mask.split(",")]
        mask_num = len(mask_files)
        mask_shape = (mask_num, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels)

        with tf.variable_scope("attack", reuse=tf.AUTO_REUSE):
            # input images
            self.clean_images = tf.placeholder(tf.float32, input_shape, name="clean_input")

            # internal parameters
            self.mask = tf.placeholder(tf.float32, mask_shape, name="mask")
            self.translate_delta = tf.placeholder(tf.float32, shape=(mask_num,2), name="translate_delta")
            self.resize_shape = tf.placeholder(tf.int32, shape=(2), name="resize_shape")
            self.homo_dest = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 4, 2), name="homo_dest")

            # external parameters
            self.brightness_delta = tf.placeholder(tf.float32, input_shape, name="brightness_delta")
            self.contrast_delta = tf.placeholder(tf.float32, name="contrast_delta")
            self.color_shifts = tf.placeholder(tf.float32, shape=input_shape, name="color_shifts")
            self.distance = tf.placeholder(tf.int32, shape=(2), name="distance")
            self.cropped_boxes = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 4), name="cropped_boxes")
            self.dest_points = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 4, 2), name="dest_points")
            self.rot_angle = tf.placeholder(dtype=tf.float32, shape=(),name="rot_angle")

            # target image
            image = generate_image().astype(np.float32)
            self.epsilon = tf.placeholder(tf.float32, shape=(), name="epsilon")
            self.target_low = tf.clip_by_value(image - self.epsilon, self.pixel_low, self.pixel_high)
            self.target_high = tf.clip_by_value(image + self.epsilon, self.pixel_low, self.pixel_high)

            # initial the patterns
            if (FLAGS.read_patterns):
                ini_patterns = np.load(FLAGS.patterns_path)
                self.patterns = tf.get_variable("patterns", dtype="float32", initializer=ini_patterns)
            else:
                random = tf.random_normal_initializer(mean=0.0, stddev=1.0)
                self.patterns = tf.get_variable("patterns", shape=image_shape, dtype="float32", initializer=random)

            # projection
            projected = tf.clip_by_value(self.patterns, self.target_low, self.target_high)
            with tf.control_dependencies([projected]):
                self.project_step = tf.assign(self.patterns, projected)

            # internal transformation
            self.perturbed_images, self.transform_patterns = internal_transform(self.clean_images, self.patterns, \
                                                      self.mask, self.translate_delta, self.resize_shape, self.homo_dest)

            # external transformation
            self.perturbed_images_transform = external_transform(self.perturbed_images, self.brightness_delta, \
                                    self.rot_angle, self.color_shifts, self.cropped_boxes, self.dest_points, self.distance)

            # pixel value constraint
            self.final_inputs = tf.clip_by_value(self.perturbed_images_transform, self.pixel_low, self.pixel_high)

    def get_pattern(self):
        return tf.clip_by_value(self.patterns, self.pixel_low, self.pixel_high)


    def build_model(self):
        assert self.final_inputs is not None
        self.fast_rcnn_inputs = self.final_inputs * 255.0

        # build the faster-rcnn model
        from lib.nets.vgg16 import vgg16
        self.net = vgg16()
        self.net.create_architecture \
            ("TEST", FLAGS.num_classes, self.fast_rcnn_inputs, tag='default', anchor_scales=[8, 16, 32])
        print('create network')

        #RPN layer
        _, _, rpn_cls_prob, _, rpn_bbox_pred, rpn_rois, top_rpn_scores, proposals = self.net._return_RPN_info()

        #Final layer
        cls_score, cls_prob, bbox_pred, final_rois = self.net._return_Scores()

        self.rpn_cls = rpn_cls_prob                                     # (1,38,38,18) for default config
        self.rpn_bbox_pred = rpn_bbox_pred                              # (1,38,38,36) for default config
        self.proposals = proposals/FLAGS.image_size                     # (12996,4) for default config -> Normalization
        self.top_rpn_scores = top_rpn_scores                            # (300,1)
        self.rpn_rois = rpn_rois[:,1:]/FLAGS.image_size                 # (300,5)-> Normalization
        self.cls_score = cls_score                                      # (300,21) for default config ~ logit
        self.cls_prob = cls_prob                                        # (300,21) for default config ~ softmax
        self.bbox_pred = bbox_pred                                      # (300,84) for default config
        self.final_rois = final_rois[:,1:]/FLAGS.image_size             # (300,5)-> Normalization

        #build model var
        self.model_vars = filter(lambda x: "attack" not in str(x.name), tf.global_variables())
        self.model_vars = set(self.model_vars) - set([self.patterns])

    def optimization(self):
        self.build_model()
        assert self.rpn_cls is not None
        assert self.cls_prob is not None

        self.rpn_target = tf.placeholder(tf.float32, shape=self.rpn_cls.shape, name="rpn_target")
        self.rpn_delta_target = tf.placeholder(tf.float32, shape=self.rpn_bbox_pred.shape, name="rpn_shape_target")
        self.object_target = tf.placeholder(tf.float32, shape=self.cls_prob.shape, name="object_target")
        self.bbox_target = tf.placeholder(tf.float32, shape=self.bbox_pred.shape, name="bbox_target")
        self.lambda_cls = tf.placeholder(tf.float32, shape=(), name="lambda_cls")
        self.lambda_reg = tf.placeholder(tf.float32, shape=(), name="lambda_reg")

        # shift the foreground bounding box shape & flip the foreground label
        # part[0] is background ; part[1] is foreground
        rpn_dimension = FLAGS.batch_size * FLAGS.rpn_size * FLAGS.rpn_size
        loss_r1 = l2_norm(self.rpn_cls - self.rpn_target)
        parts = tf.split(self.rpn_cls, num_or_size_splits=2, axis=3)
        scores = tf.reshape(parts[1], shape=[FLAGS.rpn_size * FLAGS.rpn_size * FLAGS.rpn_channel/2, 1])
        tile_scores = tf.tile(scores,[1,4])
        rpn_ctx = (self.proposals[:, 0] + self.proposals[:, 2]) / 2
        rpn_cty = (self.proposals[:, 1] + self.proposals[:, 3]) / 2
        proposals_target = tf.transpose(tf.stack([rpn_ctx, rpn_cty, rpn_ctx, rpn_cty]))
        loss_r2 = l1_norm(tile_scores * (self.proposals - proposals_target))
        if cfg.TEST.BBOX_REG:
            reshape_scores = tf.reshape(tile_scores, shape=[FLAGS.batch_size, FLAGS.rpn_size, FLAGS.rpn_size, FLAGS.rpn_channel*2])
            loss_r2 = (loss_r2 + l1_norm(reshape_scores * (self.rpn_bbox_pred - self.rpn_delta_target)))/2
        rpn_loss = FLAGS.lambda_rpn * (loss_r1 + FLAGS.lambda_balance1 * loss_r2) / rpn_dimension

        # 1. select the proposals to be flipped or distort
        # 2. distort the shape of regressor (rois+bbox)
        # 3. flip the target label of classifier
        attack_prob = tf.gather(self.cls_prob, FLAGS.attack_target, axis=1)
        self.index = tf.where(attack_prob > FLAGS.training_confidence_threshold)
        self.prob = tf.gather(self.cls_prob, self.index)
        final_ctx = (self.final_rois[:, 0] + self.final_rois[:, 2])/2
        final_cty = (self.final_rois[:, 1] + self.final_rois[:, 3])/2
        final_rois_target = tf.transpose(tf.stack([final_ctx, final_cty, final_ctx, final_cty]))
        reg_loss = self.lambda_reg * l2_norm(tf.gather(self.final_rois - final_rois_target, self.index))

        if cfg.TEST.BBOX_REG:
            target_index = [i for i in range(FLAGS.attack_target*4, (FLAGS.attack_target+1)*4)]
            pbbox = tf.gather(tf.gather(self.bbox_pred, self.index), target_index, axis=2)
            tbbox = tf.gather(tf.gather(self.bbox_target, self.index), target_index, axis=2)
            reg_loss = (reg_loss + self.lambda_reg * l2_norm(pbbox - tbbox))/2

        mean_target_prob = tf.reduce_mean(tf.gather(self.cls_prob, FLAGS.attack_target, axis=1))
        o_score = tf.gather(self.cls_score, self.index)           # ~ logits
        obj_tar = tf.gather(self.object_target, self.index)       # ~ object labels
        object_xent = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=obj_tar, logits=o_score))
        if(FLAGS.target_attack):
            cls_loss = self.lambda_cls * (mean_target_prob + FLAGS.lambda_balance2 * object_xent)
        else:
            cls_loss = self.lambda_cls * (mean_target_prob - FLAGS.lambda_balance2 * object_xent)

        # tv loss
        tv_loss = FLAGS.lambda_tv * tf.reduce_mean(tf.image.total_variation(self.patterns))

        self.total_loss = rpn_loss + cls_loss + reg_loss + tv_loss

        with tf.name_scope("adamoptimizer"):
            self.optimization_op = tf.train.AdamOptimizer().minimize(self.total_loss, var_list=[self.patterns])
        self.init_adam = tf.variables_initializer(filter(lambda x: "adam" in x.name.lower(), tf.global_variables()))
        self.init_patterns = tf.variables_initializer(set(tf.global_variables()) - set(self.model_vars))
        return self.optimization_op, self.project_step


# paste patch to target image
def generate_patch(patch, pos, target):
    x1 = int(pos[0][0] * FLAGS.image_size)
    x2 = int(pos[1][0] * FLAGS.image_size)
    y1 = int(pos[0][1] * FLAGS.image_size)
    y2 = int(pos[1][1] * FLAGS.image_size)
    target[x1:x2,y1:y2] = imresize(patch, (x2-x1,y2-y1))
    return target


# generate target image for optimization
def generate_image():
    # pre-defined images
    target_files = "face.png, upper_body.png, legs.png"

    # pre-defined area for each image
    pos = [[[0.075, 0.425], [0.2, 0.584]],
           [[0.233, 0.33], [0.5, 0.7]],
           [[0.517, 0.392], [0.875, 0.65]]]

    target_image = np.ones(shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels))*255.0/2

    image_files = [x.strip() for x in target_files.split(",")]
    for i in range(len(image_files)):
        path = os.path.join(FLAGS.target_image_dir, image_files[i])
        patch = read_preprocessed_fast_rcnn(path)
        target_image = generate_patch(patch, pos[i], target_image)
    return (target_image - cfg.PIXEL_MEANS)/255.0


# read the preprocessed image for faster-rcnn
def read_preprocessed_fast_rcnn(path):
    shape = (FLAGS.image_size, FLAGS.image_size)
    img = cv2.imread(path).astype(np.float32, copy=True)  # cv2 -> BGR
    img = (imresize(img, shape) - cfg.PIXEL_MEANS)/255.0
    return img


# transform the patterns here we utilize smaller mask to crop the patterns
def internal_transform(images, patterns, mask, translate_delta, resize_shape, homo_dest):
    n = FLAGS.batch_size
    resize_patterns = patterns

    # translate the patterns location and then combine them
    translated_mask = []
    num = mask.shape[0]
    for i in range(num):
        t_mask = tf.contrib.image.translate(mask[i],translate_delta[i])
        translated_mask.append(t_mask)
    translated_mask = tf.stack(translated_mask)
    u_mask = translated_mask[0]
    for i in range(num):
        u_mask = tf.add(u_mask,translated_mask[i]) - tf.multiply(u_mask,translated_mask[i])

    # scale the patterns size
    r_mask = tf.image.resize_images(u_mask, resize_shape)
    resize_mask = tf.image.resize_image_with_crop_or_pad(r_mask, FLAGS.image_size, FLAGS.image_size)

    # strect simulation
    src_points = \
        tf.stack([[[0., 0.], [0., FLAGS.image_size], [FLAGS.image_size, 0.], [FLAGS.image_size, FLAGS.image_size]] \
                  for _ in range(n)])
    for i in range(n):
        transforms = homography(src_points[i], homo_dest[i])
        homo_patterns = tf.contrib.image.transform(resize_patterns, transforms)
        homo_mask = tf.contrib.image.transform(resize_mask, transforms)

    patterns = tf.stack([homo_patterns] * n)
    masks = tf.stack([homo_mask] * n)
    inverse_masks = 1.0 - masks
    return images * inverse_masks + patterns * masks, homo_patterns * homo_mask


def external_transform(perturbed_images, brightness_delta, rot_angle, color_shifts, boxes, dest_points, distance):
    # recover to [0,255] for next steps
    recover_images = (perturbed_images + cfg.PIXEL_MEANS / 255.0) * 255.0

    # brightness simulation
    b_images = tf.clip_by_value(
        math_ops.add(recover_images, math_ops.cast(brightness_delta, dtypes.float32)), 0, 255)
    brightness_images = (b_images - cfg.PIXEL_MEANS) / 255.0

    # color shift
    color_images = color_shifts * brightness_images

    # distance transforms
    distance_images = tf.image.resize_image_with_crop_or_pad(
        tf.image.resize_images(color_images, size=distance), FLAGS.image_size, FLAGS.image_size)

    # camera simulation
    cropped_images = tf.image.crop_and_resize(distance_images, \
                                              boxes=boxes, box_ind=[x for x in range(FLAGS.batch_size)], \
                                              crop_size=perturbed_images.shape[1:3])

    # perspective simulation
    src_points = tf.stack([[[0., 0.], [0., FLAGS.image_size], [FLAGS.image_size, 0.],
                            [FLAGS.image_size, FLAGS.image_size]] for _ in range(FLAGS.batch_size)])
    final_images = []
    for i in range(FLAGS.batch_size):
        transforms = homography(src_points[i], dest_points[i])
        affine_image = tf.contrib.image.transform(cropped_images[i], transforms)
        rotated_image = tf.contrib.image.rotate(affine_image, rot_angle)
        final_images.append(rotated_image)

    final_images = tf.stack(final_images)
    final_images = tf.reshape(final_images, shape=perturbed_images.shape)

    return final_images


# affine
def homography(x1s, x2s):
    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    p = []
    p.append(ax(x1s[0], x2s[0]))
    p.append(ay(x1s[0], x2s[0]))

    p.append(ax(x1s[1], x2s[1]))
    p.append(ay(x1s[1], x2s[1]))

    p.append(ax(x1s[2], x2s[2]))
    p.append(ay(x1s[2], x2s[2]))

    p.append(ax(x1s[3], x2s[3]))
    p.append(ay(x1s[3], x2s[3]))

    # A is 8x8
    A = tf.stack(p, axis=0)

    m = [[x2s[0][0], x2s[0][1], x2s[1][0], x2s[1][1], x2s[2][0], x2s[2][1], x2s[3][0], x2s[3][1]]]

    # P is 8x1
    P = tf.transpose(tf.stack(m, axis=0))

    # here we solve the linear system
    # we transpose the result for convenience
    return tf.transpose(tf.matrix_solve_ls(A, P, fast=True))


# L1 norms
def l1_norm(tensor):
    return tf.reduce_sum(tf.abs(tensor))


# L2 norms
def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2)))


# read all images in dir
def read_data_fast_rcnn(folder_path):
    data = []
    filenames = os.listdir(folder_path)
    np.random.shuffle(filenames)
    fnames_new = []
    for f in filenames:
        if not f.startswith(".") and \
                (f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")):
            data.append( \
                read_preprocessed_fast_rcnn(os.path.join(folder_path, f)))
            fnames_new.append(f)
    return fnames_new, np.array(data)


# recover the image into [0,255] and write into dir
def write_reverse_preprocess_fast_rcnn(path, img):
    img = (img + cfg.PIXEL_MEANS/255.0) * 255.0
    cv2.imwrite(path, img)
