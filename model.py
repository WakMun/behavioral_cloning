import csv
import cv2
import numpy as np
import random, os
import sklearn
import tensorflow as tf
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

def makedbEntry(directory, line, samples, i, left_correction, right_correction):
    measurement = float(line[3])
        
    fname_center = directory + '/IMG/' + line[0].split('/')[-1]
    samples.append([i, fname_center, measurement, False])
    i += 1
        
    fname_center = directory + '/IMG/' + line[0].split('/')[-1]
    samples.append([i, fname_center, measurement * -1.0, True])
    i += 1
        
    fname_left = directory + '/IMG/' + line[1].split('/')[-1]
    samples.append([i, fname_left, measurement+left_correction, False])        
    i += 1
        
    fname_right = directory + '/IMG/' + line[2].split('/')[-1]
    samples.append([i, fname_right, measurement+right_correction, False])
    i += 1
    
    return i
    

def get_samples(path, left_correction, right_correction):
    '''
    reads in paths (not actual images, just paths) and measurements and shuffles them and returns this data set.
    '''
    directories = [x[0] for x in os.walk(path, followlinks=True)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    
    samples = [] #each element contains [sample_no, path, angle, inverted?]
    i=0
    for directory in dataDirectories:
        print('Collecting data from: ', directory)
        with open(directory + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None) #skip header row
            j=0
            for line in reader:
                i=makedbEntry(directory, line, samples, i, left_correction, right_correction)
                j += 1
            print('    found : ', j, ' lines')
                
    #samples = np.array(samples)
    random.shuffle(samples)
    
    return samples





def batch_maker(samples, batch_size=64):
    """
    Generates batches for training and validation.  next n_lines of input will be read and corresponding images passed to model.
    `samples` is a list of pairs (`image`, `measurement`).
    """
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for samplNo,imagePath, measurement, must_flip in batch_samples:
                originalImage = cv2.imread(imagePath)
                img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                if (must_flip):
                    #img = cv2.flip(img, 1)
                    img=np.fliplr(img)
                    
                images.append(img)
                angles.append(measurement)


            # trim image to only see section with road
            X_batch = np.array(images)
            y_batch = np.array(angles)
            yield sklearn.utils.shuffle(X_batch, y_batch)

            
def resize_normalize(x):
    import tensorflow as tf # because of this line, this function had to be declared extra instead of in a lambda. Solution from https://github.com/keras-team/keras/issues/5298
    x = tf.image.resize_images(x ,(35, 160))
    return ((x / 255.0) - 0.5)
    
    
    


samples = get_samples('./data', 0.2, -0.2)
#print (samples[0])
batch_size = 64

# Splitting samples and creating generators.
train_samples, valid_samples = train_test_split(samples, test_size=0.2)
print('=========================')
print('Total samples: {}'.format(len(samples)))
print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(valid_samples)))
steps_per_epoch = int(np.ceil(len(train_samples)/batch_size))
print ("steps_per_epoch: ",steps_per_epoch)
validation_steps = int(np.ceil(len(valid_samples)/batch_size))
print ("validation_steps: ",steps_per_epoch)
print('=========================')

train_gen = batch_maker(train_samples, batch_size=64)
valid_gen = batch_maker(valid_samples, batch_size=64)

#x, y = next(train_gen)
#print ('x:', x[0])


model = Sequential()

#Preprocessing step 1: Cropping out the clutter
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((50, 20), (0, 0))))
# Preprocessing step 2: Reduce the dimensions by a factor of 2 and normalize
#model.add(Lambda(resize_normalize, output_shape=(32, 160, 3) ))
model.add(Lambda(resize_normalize, output_shape=(35, 160, 3) ))

#Nvidia model
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
#model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(20))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

model.summary()


history_obj = model.fit_generator(train_gen, epochs=4, verbose=1,\
                                  validation_data=valid_gen, \
                                  steps_per_epoch=steps_per_epoch, \
                                  validation_steps=validation_steps)



model.save('model2.h5')

model.summary()

print(history_obj.history.keys())
print('Loss')
print(history_obj.history['loss'])
print('Validation Loss')
print(history_obj.history['val_loss'])

x = [i for i in range(1,len(history_obj.history['loss'])+1)]
print(x)

plt.plot(x, history_obj.history['loss'], '*-')
plt.plot(x, history_obj.history['val_loss'], '*-')
plt.xticks(x)
plt.title('Model Mean Squared Error Loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('images/modelloss.jpg', bbox_inches='tight')


