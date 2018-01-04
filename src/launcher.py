__author__ = 'moonkey'

import sys, argparse, logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from model.model import Model
from IPython.display import display
import exp_config
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy as sp
import scipy.ndimage
import sys
import os
import cv2

from skimage.data import page
from skimage import data, morphology, measure
from skimage.color import rgb2gray, label2rgb
from skimage.morphology import square, disk, diamond, closing, rectangle
from skimage.filters.rank import median
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from skimage.segmentation import active_contour
from skimage.measure import label, regionprops
from skimage.restoration import inpaint
from operator import itemgetter, attrgetter, methodcaller

class boundingB:
    def __init__(self, dx, dy, width, height):
        self.dx = dx
        self.dy = dy
        self.width = width
        self.height = height

def runnoword(filename = "8.jpg"):
	matplotlib.rcParams['font.size'] = 9
	image_ori = data.imread(filename)
	image = rgb2gray(image_ori)

	window_size = 15
	thresh_sauvola = threshold_sauvola(image, window_size=window_size)

	binary_sauvola = image < thresh_sauvola
	binary_sauvola = median(binary_sauvola, square(3))

	binary_sauvola_rec = morphology.dilation(binary_sauvola, rectangle(5,50))
	binary_sauvola_diamond = morphology.dilation(binary_sauvola, rectangle(5,11))
	coded_paws, num_paws = sp.ndimage.label(binary_sauvola_rec)
	data_slices = sp.ndimage.find_objects(coded_paws)
	count = 0
	count2 = 0;
	object = open("result\groundTruth.txt", "w")

	toReturn = []

	for slice in data_slices:
		dx, dy = slice
		
		result = binary_sauvola_diamond[dx.start:dx.start + dx.stop-dx.start+1, dy.start:dy.start + dy.stop-dy.start+1]
		
		coded_paws_2, num_paws_2 = sp.ndimage.label(result)
		data_slices_2 = sp.ndimage.find_objects(coded_paws_2)
		
		bB = []
		
		for slice_2 in data_slices_2:
			dx_2, dy_2 = slice_2
			x = dx.start + dx_2.start
			y = dy.start + dy_2.start
			width = dx_2.stop-dx_2.start+1
			height = dy_2.stop-dy_2.start+1
			bB.append(boundingB(x, y, width, height))
		
		bbB = sorted(bB, key=attrgetter('dy'))
		
		for i in bbB:
			result_2 = image_ori[i.dx : i.dx+i.width, i.dy : i.dy+i.height]
			result3 = binary_sauvola[i.dx : i.dx+i.width, i.dy : i.dy+i.height]

			if (i.width) * (i.height) > 900:
				boundTop = result3[0].sum()
				boundBot = result3[i.width-2].sum()
			
				if boundTop < 7 and boundBot < 8:
					toReturn.append(i)
					os.makedirs("result\\" + str(count))
					tmpStr = "result\\" + str(count) + "\\filename_" + str(count) + ".png"
					plt.imsave(tmpStr, result_2)
					object.write('.\\result\\' + str(count) + '\\filename_' + str(count) + '.png large\n');
					count = count + 1

	object.close()
	inverse = ~binary_sauvola_diamond
	idbinary_sauvola = np.dstack((inverse, inverse, inverse))
	defect = image_ori.copy()

	for i in range(idbinary_sauvola.shape[0]):
		for j in range(idbinary_sauvola.shape[1]):
			if idbinary_sauvola[i,j,0] == 0:
				defect[i,j,:] = 0
			if binary_sauvola_diamond[i,j] > 0:
				binary_sauvola_diamond[i,j] = 1

	dst_TELEA = cv2.inpaint(defect, binary_sauvola_diamond, 3, cv2.INPAINT_TELEA)
	plt.imsave("nowordpage.jpg", dst_TELEA)
	return toReturn

def process_args(args, defaults):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-id', dest="gpu_id",
                        type=int, default=defaults.GPU_ID)

    parser.add_argument('--original-picture', dest="original_picture",
                        type=str, default="9.jpg")

    parser.add_argument('--use-gru', dest='use_gru', action='store_true')

    parser.add_argument('--phase', dest="phase",
                        type=str, default=defaults.PHASE,
                        choices=['train', 'test', 'evaluate'],
                        help=('Phase of experiment, can be either' 
                            ' train or test, default=%s'%(defaults.PHASE)))
    parser.add_argument('--data-path', dest="data_path",
                        type=str, default=defaults.DATA_PATH,
                        help=('Path of file containing the path and labels'
                            ' of training or testing data, default=%s'
                            %(defaults.DATA_PATH)))
    parser.add_argument('--data-base-dir', dest="data_base_dir",
                        type=str, default=defaults.DATA_BASE_DIR,
                        help=('The base directory of the paths in the file '
                            'containing the path and labels, default=%s'
                            %(defaults.DATA_PATH)))
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help=('Visualize attentions or not'
                            ', default=%s' %(defaults.VISUALIZE)))
    parser.add_argument('--no-visualize', dest='visualize', action='store_false')
    parser.set_defaults(visualize=defaults.VISUALIZE)
    parser.add_argument('--batch-size', dest="batch_size",
                        type=int, default=defaults.BATCH_SIZE,
                        help=('Batch size, default = %s'
                            %(defaults.BATCH_SIZE)))
    parser.add_argument('--initial-learning-rate', dest="initial_learning_rate",
                        type=float, default=defaults.INITIAL_LEARNING_RATE,
                        help=('Initial learning rate, default = %s'
                            %(defaults.INITIAL_LEARNING_RATE)))
    parser.add_argument('--num-epoch', dest="num_epoch",
                        type=int, default=defaults.NUM_EPOCH,
                        help=('Number of epochs, default = %s'
                            %(defaults.NUM_EPOCH)))
    parser.add_argument('--steps-per-checkpoint', dest="steps_per_checkpoint",
                        type=int, default=defaults.STEPS_PER_CHECKPOINT,
                        help=('Checkpointing (print perplexity, save model) per'
                            ' how many steps, default = %s'
                            %(defaults.STEPS_PER_CHECKPOINT)))
    parser.add_argument('--target-vocab-size', dest="target_vocab_size",
                        type=int, default=defaults.TARGET_VOCAB_SIZE,
                        help=('Target vocabulary size, default=%s' 
                            %(defaults.TARGET_VOCAB_SIZE)))
    parser.add_argument('--model-dir', dest="model_dir",
                        type=str, default=defaults.MODEL_DIR,
                        help=('The directory for saving and loading model '
                            '(structure is not stored), '
                            'default=%s' %(defaults.MODEL_DIR)))
    parser.add_argument('--target-embedding-size', dest="target_embedding_size",
                        type=int, default=defaults.TARGET_EMBEDDING_SIZE,
                        help=('Embedding dimension for each target, default=%s' 
                            %(defaults.TARGET_EMBEDDING_SIZE)))
    parser.add_argument('--attn-num-hidden', dest="attn_num_hidden",
                        type=int, default=defaults.ATTN_NUM_HIDDEN,
                        help=('number of hidden units in attention decoder cell'
                            ', default=%s' 
                            %(defaults.ATTN_NUM_HIDDEN)))
    parser.add_argument('--attn-num-layers', dest="attn_num_layers",
                        type=int, default=defaults.ATTN_NUM_LAYERS,
                        help=('number of hidden layers in attention decoder cell'
                            ', default=%s' 
                            %(defaults.ATTN_NUM_LAYERS)))
    parser.add_argument('--load-model', dest='load_model', action='store_true',
                        help=('Load model from model-dir or not'
                            ', default=%s' %(defaults.LOAD_MODEL)))
    parser.add_argument('--no-load-model', dest='load_model', action='store_false')
    parser.set_defaults(load_model=defaults.LOAD_MODEL)
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default=defaults.LOG_PATH,
                        help=('Log file path, default=%s' 
                            %(defaults.LOG_PATH)))
    parser.add_argument('--output-dir', dest="output_dir",
                        type=str, default=defaults.OUTPUT_DIR,
                        help=('Output directory, default=%s' 
                            %(defaults.OUTPUT_DIR)))
    parser.add_argument('--max_gradient_norm', dest="max_gradient_norm",
                        type=int, default=defaults.MAX_GRADIENT_NORM,
                        help=('Clip gradients to this norm.'
                              ', default=%s'
                              % (defaults.MAX_GRADIENT_NORM)))
    parser.add_argument('--no-gradient_clipping', dest='clip_gradients', action='store_false',
                        help=('Do not perform gradient clipping, difault for clip_gradients is %s' %
                              (defaults.CLIP_GRADIENTS)))
    parser.set_defaults(clip_gradients=defaults.CLIP_GRADIENTS)

    parameters = parser.parse_args(args)
    return parameters

def main(args, defaults):
    parameters = process_args(args, defaults)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = Model(
                phase = parameters.phase,
                visualize = parameters.visualize,
                data_path = parameters.data_path,
                data_base_dir = parameters.data_base_dir,
                output_dir = parameters.output_dir,
                batch_size = parameters.batch_size,
                initial_learning_rate = parameters.initial_learning_rate,
                num_epoch = parameters.num_epoch,
                steps_per_checkpoint = parameters.steps_per_checkpoint,
                target_vocab_size = parameters.target_vocab_size, 
                model_dir = parameters.model_dir,
                target_embedding_size = parameters.target_embedding_size,
                attn_num_hidden = parameters.attn_num_hidden,
                attn_num_layers = parameters.attn_num_layers,
                clip_gradients = parameters.clip_gradients,
                max_gradient_norm = parameters.max_gradient_norm,
                load_model = parameters.load_model,
                valid_target_length = float('inf'),
                gpu_id=parameters.gpu_id,
                use_gru=parameters.use_gru,
                session = sess)
        model.launch()

def run(args, defaults):
    parameters = process_args(args, defaults)
    getList = runnoword(parameters.original_picture)
    main("", exp_config.ExpConfig)
    
    
    ann_file = open('result\\predict.txt', 'r')
    lines = ann_file.readlines()
    backgound = Image.open(".\\nowordpage.jpg")
    index =0
    outText = open('.\\FinalResults\\FinalWordOutPut.txt', 'w')
    for l in lines:
        img_path, lex = l.strip().split()
        outText.write(lex)
        if index >0 and index %10 == 0:
            outText.write('\n')
        else:
            outText.write(' ')

        ori_image = Image.open(img_path)
        fontname = '.\ToThePoint.ttf'
        gorundtruth = lex

        yscale = 1.0
        Upperflag = 0
        undercflag = 0

        if gorundtruth.islower() == False:
            yscale = 0.5
            Upperflag =1

        Findcount = gorundtruth.find("g")
        Findcount += gorundtruth.find("j")
        Findcount += gorundtruth.find("p")
        Findcount += gorundtruth.find("q")
        Findcount += gorundtruth.find("y")

        if  Findcount != -5:
            undercflag = 1
            if Upperflag ==1:
                yscale = 0.33
            else:
                yscale = 0.67

        Findcount = gorundtruth.find("b")
        Findcount += gorundtruth.find("d")
        Findcount += gorundtruth.find("f")
        Findcount += gorundtruth.find("h")
        Findcount += gorundtruth.find("k")
        Findcount += gorundtruth.find("l")
        Findcount += gorundtruth.find("t")
        # contain one of them    
        if  Findcount != -7:
            if undercflag == 1:
                yscale = 0.33
            else:
                if Upperflag == 1:
                    yscale = 0.5
                else:
                    yscale = 0.67

        print("Y scale: %.2f" % yscale)
        ori_pixel = ori_image.load()

        xpadding = 0
        flag = 0
        for i in range(ori_image.size[0]):
            if flag == 0:
                for j in range(ori_image.size[1]):
                    if flag == 0:
                        tmp = 0
                        tmp = (ori_pixel[i,j][0] + ori_pixel[i,j][1] + ori_pixel[i,j][2])/3
                        if tmp < 100:
                            flag = 1
                            print('Text color average: %.2f'% tmp)
                            xpadding = i
                    else:
                        break
            else:
                break


        # get an image
        base = Image.new('RGBA', ori_image.size, (255,255,255,0))

        # make a blank image for the text, initialized to transparent text color
        txt = Image.new('RGBA', base.size, (255,255,255,0))
        d = ImageDraw.Draw(txt)
        for i in range(200):
            tmp = ImageFont.truetype(fontname, i)
            width, height = d.textsize(gorundtruth, font=tmp)
            if height > ori_image.size[1]*(1+yscale):
                font_size = i
                break
        # get a font
        fnt = ImageFont.truetype(fontname, font_size)
        # get a drawing context
        # draw text, full opacity
        d.text((xpadding + 0,0-ori_image.size[1]*yscale-1), gorundtruth, font=fnt, fill=(0,0,0,255))

        out = Image.alpha_composite(base, txt)
        width, height = d.textsize(gorundtruth, font=fnt)
        print ('Font size: %d x %d' % (width, height))
        print('X-Padding: %d'% xpadding)
        #display(out)
        #display(ori_image)
        backgound.paste(out,(getList[index].dy,getList[index].dx),out)
        index = index +1
    backgound.save(".\\FinalResults\\FinalOutput.png")
    ann_file.close()
    outText.close()

if __name__ == "__main__":
    #main(sys.argv[1:], exp_config.ExpConfig)
    run(sys.argv[1:], exp_config.ExpConfig)

