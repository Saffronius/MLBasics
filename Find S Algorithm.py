import csv

no_attributes = 6

data_set = []

# take data as a list of lists
csvfile = open("1.csv","r")
reader = csv.reader(csvfile)
for row in reader:
    data_set.append(row)
    print(row)

init_hypothesis = [0 for i in range(no_attributes-1)]
iterations = 0
print(init_hypothesis)
for data_item in data_set:
    if data_item[-1] == "TRUE":
        if iterations == 0:
            init_hypothesis[::] = data_item[:-1]
        else:
            for i in range (len(init_hypothesis)):
                if init_hypothesis[i] != data_item[i]:
                    init_hypothesis[i] = "?"
        iterations += 1

print(init_hypothesis)
