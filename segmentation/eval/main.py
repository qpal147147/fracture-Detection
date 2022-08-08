import cv2
import argparse
import findpaths
import metrics

# https://github.com/FabioXimenes/SegmentationMetrics

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--segmented-images', default="./seg_mask", type=str)
ap.add_argument('-g', '--ground-truth', default="./gt_mask", type=str)

args = vars(ap.parse_args())

# Pick all the paths to the segmented images
try:
    segmentedPaths = list(findpaths.list_images(args['segmented_images']))
except Exception:
    print('Error on the path to the segmented images. Verify the paths carefully!')

# Pick all the paths to the ground truth images
try:
    gtPaths = list(findpaths.list_images(args['ground_truth']))
except Exception:
    print('Error on the path to the ground truth images. Verify the paths carefully!')

# Verify if the number of images are equal
if len(segmentedPaths) != len(gtPaths):
    print('The number of images to be compared must be of the same size!')
    exit()

# Create a list of the metrics that will be calculated
metrics_list = ['Jaccard', 'Matthew Correlation Coefficient', 'Dice', 'Accuracy', 'Sensitivity', 'Specificity', 
                'F1-Score', 'Precision Predictive Value', 'Negative Predictive Value', 'False Positive Rate', 'False Discovery Rate', 'False Negative Rate']

# Create a dictionary that will contain all the metrics as keys and their respective values for each image
dict_values = {}

# Load the images, compute the confusion matrix and calculate the metrics
print('Loading the images...')
for (i, images) in enumerate(zip(segmentedPaths, gtPaths)):
    print('Image {}/{}'.format(i + 1, len(segmentedPaths)))

    segmented = cv2.imread(images[0], 0)
    gt = cv2.imread(images[1], 0)

    segmented = cv2.resize(segmented, (gt.shape[1], gt.shape[0]))

    # Verify if the segmented and ground truth images have the same size
    if segmented.shape != gt.shape:
        print('The sizes of segmented image and ground truth image must be the same!')
        pass

    # Threshold the images to be sure that the pixel values are binary
    ret, segmented = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)
    ret, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)

    # Get the confusion matrix of the image
    confusion_matrix = metrics.get_confusion_matrix(segmented, gt)
    tp, tn, fp, fn = confusion_matrix
    print(f"tp, tn, fp, fn: {tp, tn, fp, fn}")

    # Calculate all the metrics and put them into a dictionary
    dict_values = metrics.get_metrics(metrics_list, dict_values, confusion_matrix)

mean_values = metrics.mean(dict_values)
print("\nMean Values:")
for metric, mean in zip(dict_values.keys(), mean_values):
        print(metric.ljust(35), '{:.5f}'.format(mean))