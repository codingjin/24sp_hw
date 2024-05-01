import csv

DATA_DIR = 'data/'

# Part 1 - misc.train.csv
labels = []
with open(DATA_DIR + 'glove/glove.train.csv', 'r') as file0:
    reader0 = csv.reader(file0)
    for row in reader0:
        labels.append(row[0])

data = []
with open(DATA_DIR + 'misc/misc-attributes-train.csv', 'r') as file1:
    reader1 = csv.reader(file1)
    for row in reader1:
        data.append(row)

# misc.train.csv
with open(DATA_DIR + 'misc/misc.train.csv', 'w', newline='') as combined_file:
    writer = csv.writer(combined_file)
    for label, row in zip(labels, data):
        writer.writerow([label] + row)


# Part 2 - misc.test.csv
labels = []
with open(DATA_DIR + 'glove/glove.test.csv', 'r') as file0:
    reader0 = csv.reader(file0)
    for row in reader0:
        labels.append(row[0])

data = []
with open(DATA_DIR + 'misc/misc-attributes-test.csv', 'r') as file1:
    reader1 = csv.reader(file1)
    for row in reader1:
        data.append(row)

# misc.test.csv
with open(DATA_DIR + 'misc/misc.test.csv', 'w', newline='') as combined_file:
    writer = csv.writer(combined_file)
    for label, row in zip(labels, data):
        writer.writerow([label] + row)


# Part 3 - misc.eval.csv
labels = []
with open(DATA_DIR + 'glove/glove.eval.anon.csv', 'r') as file0:
    reader0 = csv.reader(file0)
    for row in reader0:
        labels.append(row[0])

data = []
with open(DATA_DIR + 'misc/misc-attributes-eval.csv', 'r') as file1:
    reader1 = csv.reader(file1)
    for row in reader1:
        data.append(row)

# misc.eval.csv
with open(DATA_DIR + 'misc/misc.eval.csv', 'w', newline='') as combined_file:
    writer = csv.writer(combined_file)
    for label, row in zip(labels, data):
        writer.writerow([label] + row)





