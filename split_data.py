import os, sys,math, random, os.path
import shutil
test_size = math.ceil(0.2*145019)
valid_size = math.ceil(0.2*145019)

normal = "/media/elnaz/My Passport/backup/embedded_normal/"
simple = "/media/elnaz/My Passport/backup/embedded_simple/"

normal_test = "/media/elnaz/My Passport/backup/embedded_normal_test/"
simple_test = "/media/elnaz/My Passport/backup/embedded_simple_test/"

normal_valid = "/media/elnaz/My Passport/backup/embedded_normal_valid/"
simple_valid = "/media/elnaz/My Passport/backup/embedded_simple_valid/"

files = os.listdir(normal)
test_idx = []
dev_idx = []
while len(test_idx) < test_size:
#    files = os.listdir(normal)
#    print(len(files))
    r = random.randint(0, len(files)-1)
    if files[r] not in test_idx:
        test_idx.append(files[r])
        print(len(test_idx))


print("-----------------------------------------------------")
while len(dev_idx)<valid_size:
#    files = os.listdir(normal)
    r = random.randint(0, len(files)-1)
    if (files[r] not in test_idx) and (files[r] not in dev_idx):
        dev_idx.append(files[r])
        print(len(dev_idx))
print("***************************************************")

for f in test_idx:
    shutil.move(normal+f, normal_test+f)
    shutil.move(simple+f, simple_test+f)
#    os.remove(os.path.join(normal,f))

for f in dev_idx:
    shutil.move(normal+f, normal_valid+f)
    shutil.move(simple+f, simple_valid+f)
#    os.remove(os.path.join(normal,f))


