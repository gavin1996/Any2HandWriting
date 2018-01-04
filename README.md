# Dependencies
* Windows 10 64bit with GTX1070
* python==3.5.3
* anaconda==35
* Distance==0.1.3
* (optional)jupyter notebook
* Keras==2.0.4
* matplotlib==1.5.3
* numpy==1.11.2
* openpyxl==2.4.0
* opencv3==3.1.0
* pillow==3.4.2
* scikit-image==0.13.0
* scikit-learn==0.18.1
* scipy==0.18.1
* six==1.10.0
* tensorflow-gpu==1.0.1
* Theano==0.9.0
* tk==8.5.18
* tqdm==4.11.2
* vs2015_runtime==14.0.25123

# How to run
1. cd to root directory(the directory that contains this README.md file)
2. You have to clear all contents in **result** directory
3. run this command :
    >python src/launcher.py --original-picture=9.jpg

4. If you like to change the input image, put the image beside this README.md file and change the file name behind `--original-picture=`
5. Check for the existence of: 
    * An empty **result** directory
    * A **FinalResults** directory
    * An input image (e.g 9.jpg)
6. Do not delete/modify any unmentioned files/directories

# Final Result File (After Execution)
* In **FinalResults** directory
* There are two files: 
    * FinalWordOutPut.txt
        * Text extracted from the given image
        * We give a new line charactor every 10 words
    * FinalOutput.png
        * Modified the font of the original image to ToThePoint.ttf 