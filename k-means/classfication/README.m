write_jpg.m: loads all preprocessed images (two folders), and saves them into jpg for feature extraction.
kmeans_all.m: loops through all cases, does classfication and saves the results.
kmeans2.m/fkmeans.m: explores different classfication settings (e.g., different patch sizes, different number of classes, different number of clusters).
demo_dice.m: given the DICE coefficient of all slices and cases over different tissues, does some stastical analysis

Util functions:
zero_pad: pads the boundary of the input image with 0s based on the given size.
im2row: transforms the given 2d image into Hankle matrix form, this function is used for patch-based feature generation.
feture_process/feature_process_3d: compresses features from neural network.
kmeans_3d: given the 3d features, does classfication per slice using kmeans.
show_image: displays the results.


