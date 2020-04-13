import tensorflow as tf
import numpy as np
import os
import gc
import sys
import time
import flags
from lib.model.config import cfg
from attack_utils import *

FLAGS = flags.FLAGS

class Attack(object):
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
        config = tf.ConfigProto()
        tf.logging.set_verbosity(tf.logging.ERROR)
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

        # create session
        self.sess = tf.Session(config=config)

        # create graph
        with self.sess.as_default():
            self.attack_graph = AttackGraph()
            self.optimization_op, self.project_step = self.attack_graph.optimization()

            # initial the adam optimization
            self.sess.run(self.attack_graph.init_adam)
            self.sess.run(self.attack_graph.init_patterns)

            # load ckpt files
            fast_rcnn_model_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
            restorer = tf.train.Saver(var_list=self.attack_graph.model_vars)
            restorer.restore(self.sess, fast_rcnn_model_path)
            print('load ckpt files {:s}'.format(fast_rcnn_model_path))

            self.train_fname, self.train_data = read_data_fast_rcnn(FLAGS.attack_srcdir)

    # adaptive rePduce the epsilon value
    def _get_epsilon(self, epoch_num):
        ratio = float(epoch_num) / float(FLAGS.decay_epoch + 1e-3) * 2.0
        return FLAGS.target_image_epsilon + (1.0 - FLAGS.target_image_epsilon) * np.exp(-ratio)

    def _get_color_shifts(self, shape_of_color_shifts):
        return np.ones(shape_of_color_shifts) * np.random.uniform(FLAGS.color_shifts_min, FLAGS.color_shifts_max)

    def _get_brightness_delta(self, shape_of_brightness_delta):
        return np.ones(shape_of_brightness_delta) * np.random.uniform(FLAGS.brightness_min, FLAGS.brightness_max) * 255.0

    def _get_distance_size(self):
        rdn = np.random.uniform(FLAGS.distance_min,FLAGS.distance_max)
        return [int(FLAGS.image_size * rdn), int(FLAGS.image_size * rdn)]

    def _get_translate_delta(self):
        mask_files = [x.strip() for x in FLAGS.attack_mask.split(",")]
        mask_num = len(mask_files)
        rnds = np.random.uniform(low=FLAGS.translation_min, high=FLAGS.translation_max, size=(mask_num, 2))
        shape = [[FLAGS.mask_size, FLAGS.mask_size] for i in range(mask_num)]
        return np.array(shape * rnds).astype(int)

    def _get_resize_shape(self):
        rnds = np.random.uniform(FLAGS.resize_min, FLAGS.resize_max, size=(2))
        shape = (FLAGS.mask_size, FLAGS.mask_size) * (1.0 + rnds)
        return shape

    def _get_rotation_degree(self):
        degree = np.random.uniform(np.pi/(180/FLAGS.rotate_min), np.pi/(180/FLAGS.rotate_max))
        return degree

    def _get_boxes(self):
        return [[np.random.uniform(-0.1, 0.2), \
                np.random.uniform(-0.1, 0.2), \
                np.random.uniform(0.9, 1.2), \
                np.random.uniform(0.9, 1.2)] for _ in range(FLAGS.batch_size)]

    def _get_homo_dest(self):
        n = FLAGS.batch_size
        # source points
        src = [[[0, 0], [0, FLAGS.image_size], [FLAGS.image_size, 0],
                [FLAGS.image_size, FLAGS.image_size]] for _ in range(n)]
        offset = FLAGS.image_size * np.random.normal(loc = FLAGS.affine_mean, scale = FLAGS.affine_var, size=((n, 4, 2)))
        return src + offset

    def _get_dest_points(self):
        n = FLAGS.batch_size
        # dest points
        src = [[[0, 0], [0, FLAGS.image_size], [FLAGS.image_size, 0], [FLAGS.image_size, FLAGS.image_size]] for _ in range(n)]

        import scipy.stats as stats
        lower, upper = -FLAGS.image_size / 3, FLAGS.image_size / 3
        mu, sigma = FLAGS.transform_mean, FLAGS.transform_var
        X = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        random = X.rvs((n, 4, 2))
        return src + random

    def _get_rpn_target(self):
        rpn_label = np.zeros(shape=(FLAGS.batch_size, FLAGS.rpn_size, FLAGS.rpn_size, FLAGS.rpn_channel))
        last_dim_len = rpn_label.shape[-1]
        # set background label
        rpn_label[:, :, :, 0 : last_dim_len/2] = 1.0
        rpn_delta = -0.1 * np.ones(shape=(1, FLAGS.rpn_size, FLAGS.rpn_size, FLAGS.rpn_channel * 2))
        return rpn_label, rpn_delta

    def _get_cls_target(self):
        object_target = np.zeros(shape=(FLAGS.final_score_num, FLAGS.num_classes))
        if(FLAGS.target_attack):
            object_target[:, FLAGS.object_target] = 1.0
        else:
            object_target[:, FLAGS.attack_target] = 1.0
        _im_info = np.array([FLAGS.image_size, FLAGS.image_size, 1.0])
        return object_target, _im_info

    def _get_bbox_delta(self):
        return -0.1 * np.ones(shape=(FLAGS.final_score_num, 4 * FLAGS.num_classes))

    # training the universal camouflage patterns
    def optimize(self):
        latest_conf = 1.0
        lastest_num = 1000
        lastest_loss = 99999.0
        time1 = time.time()

        lambda_cls = FLAGS.lambda_cls
        lambda_reg = FLAGS.lambda_reg

        save_dir = os.path.join(FLAGS.save_folder, FLAGS.save_prefix)
        if(os.path.exists(save_dir)):
            print('save_dir ' + save_dir + ' is existing.')
            return
        else:
            os.makedirs(save_dir)
            print('create dir: ' + save_dir)

        for e in range(FLAGS.attack_epochs):
            if(e < FLAGS.decay_epoch):
                FLAGS.lambda_cls = 0.0
                FLAGS.lambda_reg = 0.0
            else:
                FLAGS.lambda_cls = lambda_cls
                FLAGS.lambda_reg = lambda_reg

            max_conf, remain_num, loss, patterns, transform_patterns, final_inputs = self.attacking(e)

            # save the patterns (npy)
            if max_conf < latest_conf or (remain_num < lastest_num) or loss < lastest_loss:
                latest_conf = max_conf
                lastest_num = remain_num
                lastest_loss = loss
                npy_path = os.path.join(save_dir, "patterns-epoch-%04d.npy" % e)
                np.save(npy_path, patterns)

            # save the images
            if patterns is not None:
                time2 = time.time()
                print('Total time: {}'.format(time2 - time1))
                write_reverse_preprocess_fast_rcnn( \
                    os.path.join(save_dir, "patterns-epoch-%04d.png" % e), patterns)

            if transform_patterns is not None:
                write_reverse_preprocess_fast_rcnn( \
                    os.path.join(save_dir, "transform-patterns-epoch-%04d.png" % e), transform_patterns)

            if final_inputs is not None:
                write_reverse_preprocess_fast_rcnn( \
                    os.path.join(save_dir, "final-inputs-epoch-%04d.png" % e), final_inputs)

            succ_rate = 1 - float(remain_num) / float(len(self.train_data))
            if( succ_rate <= 1.0 and succ_rate > FLAGS.stop_threshold):
                print("early stop")
                return ;


    # set feed data
    def create_feed_dict(self, data, attack_graph, epoch):
        rpn_label, rpn_delta = self._get_rpn_target()
        object_target, _im_info = self._get_cls_target()
        bbox_delta = self._get_bbox_delta()

        feed_dict = {
            attack_graph.clean_images: data, \
            attack_graph.epsilon: self._get_epsilon(epoch), \
            attack_graph.mask: get_mask_files() / 255.0, \
            attack_graph.homo_dest: self._get_homo_dest(),\
            attack_graph.color_shifts: self._get_color_shifts(data.shape), \
            attack_graph.cropped_boxes: self._get_boxes(), \
            attack_graph.dest_points: self._get_dest_points(), \
            attack_graph.distance: self._get_distance_size(), \
            attack_graph.translate_delta: self._get_translate_delta(), \
            attack_graph.resize_shape:self._get_resize_shape(),\
            attack_graph.brightness_delta:self._get_brightness_delta(data.shape), \
            attack_graph.rot_angle: self._get_rotation_degree(), \
            attack_graph.rpn_target: rpn_label, \
            attack_graph.object_target: object_target, \
            attack_graph.lambda_cls: FLAGS.lambda_cls, \
            attack_graph.lambda_reg: FLAGS.lambda_reg, \
            attack_graph.net._im_info: _im_info
            }

        if cfg.TEST.BBOX_REG:
            feed = {}
            feed[attack_graph.rpn_delta_target] = rpn_delta
            feed[attack_graph.bbox_target] = bbox_delta
            feed_dict.update(feed)

        return feed_dict

    # train the pattern or each epoch
    def attacking(self, epoch_num):
        assert self.train_data is not None
        n_imgs = len(self.train_data)
        batch_size = FLAGS.batch_size
        num_batches = n_imgs / batch_size

        report = (epoch_num % FLAGS.save_freq == 0)

        if report:
            avg_loss = 0.0
        else:
            avg_loss = 10000000.0

        max_conf_sum = 0.0
        r_num = 0
        over_max_num = 0

        # begin to train the patterns
        for b in range(0, n_imgs, batch_size):
            curr_data = self.train_data[b: b + batch_size]
            curr_name = self.train_fname[b: b + batch_size]

            feed_dict = self.create_feed_dict(np.array(curr_data), self.attack_graph, epoch_num)

            if(report):
                ops = [self.optimization_op, self.attack_graph.total_loss]
            else:
                ops = self.optimization_op

            results = self.sess.run(ops, feed_dict=feed_dict)

            # run the projection
            if(FLAGS.target_image):
                self.sess.run(self.project_step, feed_dict=feed_dict)

            if(report):
                avg_loss += results[1]

                # The proposal numbers which detected as attacked label
                pred_index , pred_vec = self.sess.run([self.attack_graph.index, self.attack_graph.prob],
                    feed_dict=feed_dict)
                if(len(pred_index) > 0):
                    max_conf_sum += np.max(pred_vec)
                    r_num += 1
                    if (np.max(pred_vec) > FLAGS.testing_confidence_threshold):
                        over_max_num += 1

        if (report):
            avg_loss = avg_loss / float(num_batches)

            report_string = "Epoch %d" % epoch_num
            report_string += " [train] total_loss %.4f" % avg_loss

            max_conf = 1.0
            if(r_num > 0):
                max_conf = max_conf_sum / float(r_num)
                report_string += " avg max_confidence %.4f" % (max_conf)
            report_string += " remain_attack_images %.3f" % (float(r_num) / float(len(self.train_data)))
            report_string += " over_max_num %.3f" % (float(over_max_num) / float(len(self.train_data)))

            print(report_string)

            # visualization the result
            data_len = len(self.train_data)
            index_start = np.random.randint(0, data_len - batch_size)
            curr_data = self.train_data[index_start: index_start + batch_size]
            feed_dict = self.create_feed_dict(np.array(curr_data), self.attack_graph, epoch_num)
            index = np.random.randint(0, batch_size)
            patterns, transform_patterns, final_inputs = self.sess.run( \
                [self.attack_graph.get_pattern(), \
                 self.attack_graph.transform_patterns,\
                 self.attack_graph.final_inputs[index]], \
                feed_dict=feed_dict)

        else:
            patterns = None
            transform_patterns = None
            final_inputs = None
            max_conf = 1.0
            r_num = 10000

        feed_dict = None
        gc.collect()
        return max_conf, r_num, avg_loss, patterns, transform_patterns, final_inputs


def get_mask_files():
    if FLAGS.attack_mask != "":
        mask_names = [x.strip() for x in FLAGS.attack_mask.split(",")]
    else:
        mask_names = []

    mask_files = []
    for i in range(len(mask_names)):
        mask_path = os.path.join(FLAGS.mask_dir,mask_names[i])
        mask = cv2.imread(mask_path).astype(np.float32, copy=True)
        mask_files.append(mask)

    return np.array(mask_files)
