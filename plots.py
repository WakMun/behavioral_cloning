import matplotlib.pyplot as plt
import os
import csv
import numpy as np 
import cv2

def get_sample(path, left_correction, right_correction):    
    samples = [] #each element contains [sample_no, path, angle, inverted?]
    angles = []
    print('Collecting data from: ', path)
    with open(path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) #skip header row
        for line in reader:
            angle = 25*float(line[3])
            angles.append(angle)
            
            if (angle > 20):
                #left image
                imagePath = path + '/IMG/' + line[1].split('/')[-1]
                img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                samples.append([angle+(left_correction*25), img])
                
                #center image
                imagePath = path + '/IMG/' + line[0].split('/')[-1]
                img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                samples.append([angle, img])
                
                #right image
                imagePath = path + '/IMG/' + line[2].split('/')[-1]
                img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                samples.append([angle+(right_correction*25), img])
                
                #center flipped image
                imagePath = path + '/IMG/' + line[0].split('/')[-1]
                img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                img=np.fliplr(img)
                samples.append([angle*-1, img])
                
                break
    #print (max([abs(i) for i in angles]))                   
    return samples




samples = get_sample('./data/data',0.2,-0.2)
samples = np.array(samples)
print(samples.shape)

fig, axs = plt.subplots(2,3, figsize=(15, 6))
fig.suptitle("Images used for Training", fontsize=15)
label = ['Left Camera', 'Center Camera', 'Right Camera','Center Flipped']


axs[0,0].imshow(samples[0][1])
axs[0,0].set_title("{0} \nSteering angle:{1}".format(label[0],samples[0][0]))

axs[0,1].imshow(samples[1][1])
axs[0,1].set_title("{0} \nSteering angle:{1}".format(label[1],samples[1][0]))

axs[0,2].imshow(samples[2][1])
axs[0,2].set_title("{0} \nSteering angle:{1}".format(label[2],samples[2][0]))

axs[1,1].imshow(samples[3][1])
axs[1,1].set_title("{0} \nSteering angle:{1}".format(label[3],samples[3][0]))

axs[1,0].set_visible(False)
axs[1,2].set_visible(False)
for ax in axs.flat:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('images/trainingImages.jpg', bbox_inches='tight')
