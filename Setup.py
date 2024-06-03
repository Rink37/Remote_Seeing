import os

root = os.path.normpath(os.path.dirname(os.path.abspath(__file__))) #This is the directory containing the setup.py file

#If default folders don't exist, create them 
folderheadings = [r'\Images', r'\Programs', r'\SaveData', r'\Figures']
for h in folderheadings:
    if not os.path.exists(root+h):
        os.makedirs(root+h)
    else:
        print(root+h, 'already exists')

#Moves python programs from root folder to correct location
programnames = [r'\Camera_Comm.py', r'\CC_calc.py', r'\Data_extractor.py', r'\Data_processor.py', r'\FE_Fns.py', r'\Image_processor.py', r'\Model_fitter.py', r'\Plotter.py', r'\Temp_Comm.py', r'\Testbed.py']
for p in programnames:
    if not os.path.exists(root+p):
        if not os.path.exists(root+r'\Programs'+p):
            print('File', p, 'not found')
        else:
            print('File', p, 'is already in the correct place')
    else:
        os.replace(root+p, root+r'\Programs'+p)
        
#Store the root in a .txt file for programs to read
with open(root+r'\Programs\root.txt', 'w') as f:
    f.write(root)