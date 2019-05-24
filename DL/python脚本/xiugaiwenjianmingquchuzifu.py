import os

def batchrename(path):
    os.chdir(path)
    print("   CHANGE DIRECTORY TO THE PATH   ")
    filelist = os.listdir(path)
    print("   File List   ")
    for filename in filelist:
        print(filename)
        if "muchong.com" in filename:
            print("-----------------------This one needs renamed")
            pos = filename.find("]")
            newname = filename[pos+1:]
            os.rename(filename,newname)
  
batchrename("C:\Users\zhanchao\Downloads")