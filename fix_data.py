import csv
import os

filename = "train_62classes.csv"
# open csv file and modify each row
# save the modified file as train_62classes_fixed.csv
with open(filename, "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    with open("train_62classes_fixed.csv", "w") as csvfile2:
        writer = csv.writer(csvfile2, delimiter=",")
        for row in reader:
            path = row[1]
            # grab parent directory
            parent = os.path.dirname(path)
            filename = os.path.basename(path)
            # grab name of parent directory
            parent_name = os.path.basename(parent)
            parent_name_new = "_".join(parent_name.split("_")[:-1]) + "_" + "-1"
            # create new path
            path_new = os.path.join(parent, parent_name_new, filename)
            print(path_new)
            # write new path to csv file
            row[1] = path_new
            writer.writerow(row)
