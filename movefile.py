import os
import shutil

target_dir = r"C:\Users\nihar\Desktop"


# Print all the file in the source folder
#print(os.listdir(source_dir))

# Print different extensions of the file
extensions ={item.split('.')[-1] for item in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir,item))}
#print(extensions)


# create folder for each extension type
for extension in extensions:
    if not os.path.exists(os.path.join(target_dir,extension)):
       os.mkdir(os.path.join(target_dir,extension))

# Move files
for item in os.listdir(target_dir):
    if os.path.isfile(os.path.join(target_dir,item)):
        file_extension = item.split('.')[-1]
        shutil.move(os.path.join(target_dir,item),os.path.join(target_dir,file_extension,item))
