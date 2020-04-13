from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
# Training setup
flags.DEFINE_integer("attack_epochs", 2000, "How many iterations to run the attack for")

flags.DEFINE_integer("batch_size", 1, "The batch size to use in the attack")

flags.DEFINE_integer('image_size', 600, 'The size of each input images')

flags.DEFINE_integer("image_channels", 3, "Number of channels in input image")

flags.DEFINE_integer('num_classes', 21, 'The number of classes this network is trained on')

flags.DEFINE_integer("save_freq", 25, "How often to save")

flags.DEFINE_boolean("read_patterns", False, "Whether read the patterns(.npy) from files")

flags.DEFINE_string("patterns_path", "" , "The dir of stored patterns(.npy)")

flags.DEFINE_string('attack_srcdir', 'data/person/train/', 'The dir containing the images to train the patterns')

# Target attack setup
flags.DEFINE_boolean("target_attack", True, "True:target attack; False: untarget attack")

flags.DEFINE_integer('attack_target', 15, 'The class being hided, e.g 15->person')

flags.DEFINE_integer('object_target', 12, 'The class being targeted for this attack as a number, e.g 12->dog')

# Masks Setup
flags.DEFINE_string("mask_dir", "masks/person_mask/", "The dir of masks for generating patterns")

flags.DEFINE_string("attack_mask", "person_face.png, person_chest.png, person_hand1.png, person_hand2.png, \
                                    person_uleg1.png, person_uleg2.png, person_dleg1.png, person_dleg2.png",
                     "The name list of masks for generating patterns")

flags.DEFINE_integer("mask_size", 600, "The size of each mask")


# Internal simulation setup
flags.DEFINE_boolean("target_image", True, "True: use the natural image as constraint")

flags.DEFINE_string("target_image_dir", "images/", "The target dir for optimization")

flags.DEFINE_float("target_image_epsilon", 0.15, "The modify epsilon of pattern")

flags.DEFINE_float("resize_max", 0.05, "The hyper-parameter of pattern resizing")

flags.DEFINE_float("resize_min", -0.05, "The hyper-parameter of pattern resizing")

flags.DEFINE_float("affine_mean", 0.0, "The hyper-parameter of pattern deformation")

flags.DEFINE_float("affine_var", 0.03, "The hyper-parameter of pattern deformation")

flags.DEFINE_float("translation_max", 0.04, "The hyper-parameter of pattern translation")

flags.DEFINE_float("translation_min", -0.04, "The hyper-parameter of pattern translation")

# External simulation setup
flags.DEFINE_float("transform_mean", 0.0, "The hyper-parameter of the destination points offset when generating the perspective transforms")

flags.DEFINE_float("transform_var", 0.05, "The hyper-parameter of the destination points offset when generating the perspective transforms")

flags.DEFINE_float("distance_max", 1.5, "The hyper-parameter of distance simulation")

flags.DEFINE_float("distance_min", 0.15, "The hyper-parameter of distance simulation")

flags.DEFINE_float("color_shifts_min", 0.1, "The hyper-parameter of color shifts")

flags.DEFINE_float("color_shifts_max", 1.0, "The hyper-parameter of color shifts")

flags.DEFINE_float("brightness_max", 0.25, "The hyper-parameter of illumination control")

flags.DEFINE_float("brightness_min", -0.25, "The hyper-parameter of illumination control")

flags.DEFINE_float("rotate_max", 20, "The hyper-parameter of camera simulation")

flags.DEFINE_float("rotate_min", -20, "The hyper-parameter of camera simulation")

# Faster-RCNN setup
flags.DEFINE_string("model_dir", "", "The dir of vgg16 based faster-rcnn")

flags.DEFINE_string("model_name", "", "The ckpt of vgg16 based faster-rcnn")

flags.DEFINE_integer("rpn_size", 38, "The size of rpn features, i.e. 38 correspond to the image size 600")

flags.DEFINE_integer("rpn_channel", 18, "The channel of rpn features")

flags.DEFINE_integer("final_score_num", 300, "The num of final detections -> faster-rcnn config files")

flags.DEFINE_float("training_confidence_threshold", 0.3, "The threshold of detection in training step")

flags.DEFINE_float("testing_confidence_threshold", 0.5, "The threshold of detection in testing step")

flags.DEFINE_float("stop_threshold", 0.98, "The threshold of stopping training")

flags.DEFINE_integer("decay_epoch", 100, "The epoch of first training stage")

# Hyper-parameters
flags.DEFINE_float("lambda_balance1", 0.005, "The balance parameter")

flags.DEFINE_float("lambda_balance2", 0.002, "The balance parameter")

flags.DEFINE_float("lambda_rpn", 100.0, "The balance parameter of rpn loss")

flags.DEFINE_float("lambda_cls", 1000.0, "The balance parameter of cls loss")

flags.DEFINE_float("lambda_reg", 10.0, "The balance parameter of reg loss")

flags.DEFINE_float("lambda_tv", 0.0001, "The balance parameter of tv loss")

# System setup
flags.DEFINE_string("save_folder", "./results/", "The folder for saving the checkpoints and noise images")

flags.DEFINE_string("save_prefix", "person", "The prefix for saving checkpoints and noise images")

flags.DEFINE_string("gpu", "0", "The index of trained GPU")

flags.DEFINE_float("gpu_memory_fraction", 0.7, "The percentage fraction of used GPU")


def print_attack_flags():
    print("Parameters")
    for k in sorted(FLAGS):
        print("%s: %s"%(k, FLAGS[k].value))

