import os

base_path = "D:/PyCharmProjects/signlanguagetranslate(test)/RawData/Alphabets/"

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

for label in labels:
    path = os.path.join(base_path, label)

    if os.path.exists(path):
        for fileName in os.listdir(path):
            filePath = os.path.join(path, fileName)
            try:
                if os.path.isfile(filePath):
                    os.remove(filePath)
                    print("removed")
            except Exception as e:
                print(e)
    else:
        print("folder does not exist")
