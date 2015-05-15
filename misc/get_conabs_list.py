infp = open("concreteness_ratings.csv")
infp.readline()
concrete = []
abstract = []
ABS_THRESH = 2.0
CON_THRESH = 4.0
for line in infp:
    line_info = line.split(",")
    if float(line_info[2]) >= CON_THRESH:
        concrete.append(line_info[0])
    elif float(line_info[2]) <= ABS_THRESH:
        abstract.append(line_info[0])

consfp = open("concretewords.txt", "w")
for word in concrete:
    consfp.write(word + "\n")
absfp = open("abstractwords.txt", "w")
for word in abstract:
    absfp.write(word + "\n")
