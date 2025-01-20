#
# split a dataset for classifier learning
#
import argparse, os, random, shutil


# Split dataset
def classifier_split(dataset_dir, force_write, train_dir, test_dir, split_prob, split_number):
    """
    Create a split of the data for classification.  Split the images into training and cross-validation
    sets.  Make sure at least 1 training image remains per class.
    """
    # if we are forcing writing data, remove the current train and test directories.
    # if we are NOT forcing writing data AND the training directory exists, return.
    if os.path.exists(train_dir):
        if not force_write:
            return
        shutil.rmtree(train_dir)

    # If the test directory exists remove it because we are going to
    # re-write both the training and test datasets
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    parent_dir = os.path.dirname(train_dir)
    os.makedirs(parent_dir, exist_ok=True)
    os.mkdir(train_dir)
    os.mkdir(test_dir)

    classes = []
    images = []
    filename_to_class = {}
    fpath_to_class = {}
    class_to_remaining_images = {}
    for classname in os.listdir(dataset_dir):
        cpath = os.path.join(dataset_dir, classname)
        if os.path.isdir(cpath):
            classes.append(classname)

            # Now add to list of all images (for classification) along with map from
            # file name to class name so we can make sure we do not put all images
            # from a class into cross validation set.  TODO - if really get serious
            # will also need a test set.

            for filename in os.listdir(cpath):
                if not os.path.isdir(filename) and filename not in filename_to_class:
                    filename_to_class[filename] = classname
                    class_to_remaining_images.setdefault(classname, 0)
                    class_to_remaining_images[classname] += 1
                    fpath = os.path.join(cpath, filename)
                    fpath_to_class[fpath] = classname
                    images.append((fpath, filename))

    if split_number <= 0.0:
        assert split_prob > 0, "split prob {} must be > 0 if split number < 0!".format(split_prob)
        split_number = int(split_prob * len(images))

    # Make sure each class name has a folder in test and train
    created_dirs = set()
    for classname in classes:
        src_class_dir = os.path.join(train_dir, classname)
        dst_class_dir = os.path.join(test_dir, classname)
        if dst_class_dir not in created_dirs:
            os.mkdir(src_class_dir)
            os.mkdir(dst_class_dir)
        created_dirs.add(dst_class_dir)
    
    # Random shuffle takes the first split_number as test images with remaining images for training
    random.shuffle(images)
    n_images = 1
    for image_path, image_name in images:
        classname = fpath_to_class[image_path]
        if n_images <= split_number:
            # first see if we can remove this image from test.  We require at least
            # 1 test image in a class
            if class_to_remaining_images[classname] < 2:
                continue
            class_to_remaining_images[classname] -= 1
            # write into test_dir
            dst_class_dir = os.path.join(test_dir, classname)
        else:
            # write into train_dir
            dst_class_dir = os.path.join(train_dir, classname)

        n_images += 1

        # copy image to the destination class directory
        # loop over dat and write into class_dir
        base_image_name, fext = os.path.splitext(image_name)
        if not fext:
            img_file_name = "{}.jpeg".format(base_image_name)
        else:
            img_file_name = image_name

        dst_image_path = os.path.join(dst_class_dir, img_file_name)
        # print("idx {} cp {} -> {}".format(inst_class, src_image_path, dst_image_path))
        shutil.copy(image_path, dst_image_path)
        if n_images % 1000 == 0:
            print("processed {} iids".format(n_images))


# SETTINGS
parser = argparse.ArgumentParser(description='classifier_split')
parser.add_argument('data', help='path to complete dataset')
parser.add_argument('-t', '--target', type=str, required=True,
                    help="target data directory")
parser.add_argument('-s', '--split-num', default=-1, type=int,
                    help='number of cross validation examples')
parser.add_argument('--prob', default=0.10, type=float,
                    help='Split using probability instead of number of cross validation examples')
parser.add_argument('-f', '--force', default=False, action='store_true',
                    help='force overwrite of train/test')

args = parser.parse_args()

train_dir = os.path.join(args.target, "train")
test_dir = os.path.join(args.target, "test")
classifier_split(args.data, args.force, train_dir, test_dir, args.prob, args.split_num)