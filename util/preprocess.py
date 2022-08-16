import glob
import os
import argparse

import shutil
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut
from sklearn.model_selection import KFold

# rename dicom file in dir to special format
def rename(in_dir):
    file_list = glob.glob(in_dir + "/*.*")

    for index, path in enumerate(file_list):
        ds = dcmread(path)
        old_name = path
        new_name = in_dir + "\\" + ds.Modality + "_" + str(ds.SeriesNumber) + "_" + ds.SOPInstanceUID + ".dcm"
        os.rename(old_name, new_name)

        print(index, end="\r")

# convert dicom to bmp image
def dicom_to_image(in_dir, out_dir, HU=False):
    file_list = glob.glob(in_dir + "/*.dcm")

    for index, path in enumerate(file_list):
        ds = dcmread(path)
        file_name = ds.Modality + "_" + ds.SOPInstanceUID

        if HU:
            hu_image = apply_modality_lut(ds.pixel_array, ds)
            vmin = ds.WindowCenter - ds.WindowWidth/2
            vmax = ds.WindowCenter + ds.WindowWidth/2

            plt.imsave((out_dir + "/" + file_name + ".bmp"), hu_image, cmap='gray', vmin=vmin, vmax=vmax)
        else:
            plt.imsave((out_dir + "/" + file_name + ".bmp"), ds.pixel_array, cmap='gray')

        print(index+1, end="\r")

# search for the same file in a dir
def search_same_file(dir1, dir2, out_dir):
    # all files
    dir1_name_list = os.listdir(dir1 + "/")

    # target
    dir2_name_list = os.listdir(dir2 + "/")

    repeat_file = set(dir1_name_list) & set(dir2_name_list)
    for file in repeat_file:
        source_file = dir1 + "/" + file
        destination_file = out_dir + "/" + file
        shutil.copyfile(source_file, destination_file)
        
        print(destination_file)
    
    print(f"Found file number: {len(repeat_file)}")
    print("Output Dir: ", out_dir)

# kfold for YOLO
def spilt_img_KFold(img_dir, label_dir, out_dir, k=3):
    img_list = np.asarray(glob.glob(img_dir + "/*.*"))
    label_list = np.asarray(glob.glob(label_dir + "/*.*"))
    print(f"image number: {len(img_list)}\nlabel number: {len(label_list)}\n")

    for idx, (train_idx, test_idx) in enumerate(KFold(n_splits=3, shuffle=True, random_state=1122).split(img_list)):
        # create output dir
        for i in ["images", "labels"]:
            for j in ["train", "val", "test"]:
                root = f"kfold_{idx+1}/{i}/{j}"
                os.makedirs(os.path.join(out_dir, root), exist_ok=True)
        
        # spilt data
        train = train_idx[:int(len(train_idx)*(k-1)/k)]
        val = train_idx[int(len(train_idx)*(k-1)/k)::]
        test = test_idx
        print(f"kfold {idx+1}:\ntrain number: {len(train)}\nval number: {len(val)}\ntest number: {len(test)}\n")

        # save data
        for i in ["images", "labels"]:
            # training data
            for j in train:
                root = f"kfold_{idx+1}/{i}/train/"
                filename = img_list[j].split("\\")[-1][:-3]
                target = os.path.join(os.path.join(out_dir, root), filename)
                
                if i == "images": shutil.copyfile(img_list[j], target+"bmp")
                else:
                    source = os.path.join(label_dir, filename+"txt")
                    if source in label_list:
                        shutil.copyfile(source, target+"txt")

            # val data
            for j in val:
                root = f"kfold_{idx+1}/{i}/val/"
                filename = img_list[j].split("\\")[-1][:-3]
                target = os.path.join(os.path.join(out_dir, root), filename)
                
                if i == "images": shutil.copyfile(img_list[j], target+"bmp")
                else:
                    source = os.path.join(label_dir, filename+"txt")
                    if source in label_list:
                        shutil.copyfile(source, target+"txt")

            # test data
            for j in test:
                root = f"kfold_{idx+1}/{i}/test/"
                filename = img_list[j].split("\\")[-1][:-3]
                target = os.path.join(os.path.join(out_dir, root), filename)
                
                if i == "images": shutil.copyfile(img_list[j], target+"bmp")
                else:
                    source = os.path.join(label_dir, filename+"txt")
                    if source in label_list:
                        shutil.copyfile(source, target+"txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="argument for different function", dest="function")
    parser_rename = subparsers.add_parser("rename", help="rename dicom file in dir to special format")
    parser_rename.add_argument("dir", type=str, help="input dir")

    parser_kfold = subparsers.add_parser("kfold", help="kfold for YOLO")
    parser_kfold.add_argument("img", type=str, help="img dir path")
    parser_kfold.add_argument("label", type=str, help="label dir path")
    parser_kfold.add_argument("output", type=str, help="save to output path")

    parser_search = subparsers.add_parser("search-file", help="search for the same file in dir")
    parser_search.add_argument("source", type=str, help="dir to search")
    parser_search.add_argument("target", type=str, help="target dir to search")
    parser_search.add_argument("output", type=str, help="save to output path")

    parser_dicom = subparsers.add_parser("dicom2img", help="convert dicom to bmp image")
    parser_dicom.add_argument("source", type=str, help="dicom dir path")
    parser_dicom.add_argument("output", type=str, help="save to output path")
    parser_dicom.add_argument("--HU", action="store_true", help="convert to HU image")

    opt = parser.parse_args()
    print(opt)

    if opt.function == "rename":
        rename(in_dir=opt.dir)
    elif opt.function == "kfold":
        spilt_img_KFold(img_dir=opt.img, label_dir=opt.label, out_dir=opt.output)
    elif opt.function == "search-file":
        search_same_file(dir1=opt.source, dir2=opt.target, out_dir=opt.output)
    elif opt.function == "dicom2img":
        dicom_to_image(in_dir=opt.source, out_dir=opt.output, HU=opt.HU)