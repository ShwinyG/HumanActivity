import socket
import sys
import json
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import scipy as sp
from scipy.ndimage.interpolation import shift
import csv

count = 0


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import socket
import sys
import json
import threading
import time


user_id = "grp"
# TODO: list the class labels that you collected data for in the order of label_index (defined in collect-labelled-data.py)
class_names = ["walking", "running", "sitting", "standing"] #...


def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window.
    """
    return np.mean(window, axis=0)

# TODO: define functions to compute more features

def _compute_amax_features(window):
    return np.amax(window, axis=0)

def _compute_median_features(window):
    return np.median(window, axis=0)

def _compute_variance_features(window):
    return np.var(window, axis=0)

def extract_features(window):
    """
    Here is where you will extract your features from the data over
    the given window. We have given you an example of computing
    the mean and appending it to the feature vector.

    """
    x = []
    feature_names = []

    x.append(_compute_mean_features(window))
    x.append(_compute_amax_features(window))
    x.append(_compute_median_features(window))
    x.append(_compute_variance_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")
    feature_names.append("x_amax")
    feature_names.append("y_amax")
    feature_names.append("z_amax")
    feature_names.append("x_median")
    feature_names.append("y_median")
    feature_names.append("z_median")
    feature_names.append("x_variance")
    feature_names.append("y_variance")
    feature_names.append("z_variance")


    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector

def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable.
    Thanks to https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/"""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("*ERROR* sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("*ERROR* type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("*ERROR* step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("*ERROR* winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence)-winSize)/step)+1

    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield i, sequence[i:i+winSize]


GRAVITY = 9.81
READ_LIMIT = 400;
acc_readings = np.zeros((READ_LIMIT, 3))

acc_state = False;
read_counter = 0;
aggX = 0
aggY = 0
aggZ = 0

def reset_vars():
    """
    Resets the variables used in reorientation. Since they are global
    variables, we need to make sure that they are reset. In the future,
    this should really be done using some sort of Python object.
    """

    global acc_state
    global read_counter
    global aggX
    global aggY
    global aggZ

    acc_state = False;
    read_counter = 0;
    aggX = 0
    aggY = 0
    aggZ = 0

def reorient(acc_x, acc_y, acc_z):
    """
    Reorients the accelerometer data. It comes from some legacy
    Java code, so it's very messy. You don't need to worry about
    how it works.
    """
    x = acc_x
    y = acc_z
    z = -acc_y

    global acc_state
    global read_counter
    global aggX
    global aggY
    global aggZ

    if read_counter >= READ_LIMIT:
        read_counter = 0

    accState = True;

    aggX += x - acc_readings[read_counter][0];
    aggY += y - acc_readings[read_counter][1];
    aggZ += z - acc_readings[read_counter][2];

    acc_readings[read_counter][0] = x;
    acc_readings[read_counter][1] = y;
    acc_readings[read_counter][2] = z;

    if(accState):
        acc_z_o = aggZ/(READ_LIMIT*GRAVITY);
        acc_y_o = aggY/(READ_LIMIT*GRAVITY);
        acc_x_o = aggX/(READ_LIMIT*GRAVITY);

        if acc_z_o > 1.0:
            acc_z_o = 1.0
        if acc_z_o < -1.0:
            acc_z_o = -1.0
        x = x/GRAVITY;
        y = y/GRAVITY;
        z = z/GRAVITY;

        theta_tilt = np.arccos(acc_z_o);
        phi_pre = np.arctan2(acc_y_o, acc_x_o);
        tan_psi = (-acc_x_o*np.sin(phi_pre) + acc_y_o*np.cos(phi_pre))/((acc_x_o*np.cos(phi_pre)+acc_y_o*np.sin(phi_pre))*np.cos(theta_tilt)-acc_z_o*np.sin(theta_tilt));
        psi_post = np.arctan(tan_psi);
        acc_x_pre = x*np.cos(phi_pre)+ y*np.sin(phi_pre);
        acc_y_pre = -x*np.sin(phi_pre)+ y*np.cos(phi_pre);
        acc_x_pre_tilt = acc_x_pre*np.cos(theta_tilt)-z*np.sin(theta_tilt);
        acc_y_pre_tilt = acc_y_pre;
        orient_acc_x = (acc_x_pre_tilt*np.cos(psi_post)+acc_y_pre_tilt*np.sin(psi_post))*GRAVITY;
        orient_acc_y =(-acc_x_pre_tilt*np.sin(psi_post)+acc_y_pre_tilt*np.cos(psi_post))*GRAVITY;
        orient_acc_z = z*GRAVITY/(np.cos(theta_tilt));

        if orient_acc_z > 3 * GRAVITY:
            orient_acc_z = 3 * GRAVITY;
        if orient_acc_z < -3 * GRAVITY:
            orient_acc_z = -3 * GRAVITY;

        orient_acc_z = np.sqrt((x*x+y*y+z*z)*GRAVITY*GRAVITY - (orient_acc_x*orient_acc_x + orient_acc_y*orient_acc_y));

        result = [orient_acc_x, orient_acc_y, orient_acc_z]
    read_counter += 1;
    return result;


data_file = '/Users/mehulramaswami/OneDrive/Documents/Umass/Sem 7/CS 328/final-new-1.csv'
data = np.genfromtxt(data_file, delimiter=',')

new_data_file = 'partial-data.csv'

with open(new_data_file, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data[10190:11190])
    writer.writerows(data[24850:26800])
    writer.writerows(data[:1050])
    writer.writerows(data[10190:11190])
visualize_data = np.genfromtxt(new_data_file, delimiter=',')
visualize_data[:,[4]] = visualize_data[:,[4]].astype(int)


reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

window_size = 20
step_size = 20

# sampling rate should be about 25 Hz; you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

X = []; Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])

X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

cv = KFold(n_splits=10, random_state=None, shuffle=True)
avg_acc = 0; avg_prec = 0; avg_rec = 0
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]


    """
    TODO: iterating over each fold, fit a decision tree classifier on the training set.
    Then predict the class labels for the test set and compute the confusion matrix
    using predicted labels and ground truth values. Print the accuracy, precision and recall
    for each fold.
    """
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    tree = tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    # TODO: calculate and print the average accuracy, precision and recall values over all 10 folds
    avg_acc += metrics.accuracy_score(y_test, y_pred)
    avg_prec += metrics.precision_score(y_test, y_pred, average='micro')
    avg_rec += metrics.recall_score(y_test, y_pred, average='micro')
    print(metrics.confusion_matrix(y_test, y_pred))
print("Average accuracy: ", avg_acc / 10)
print("Average precision: ", avg_prec / 10)
print("Average recall: ", avg_rec / 10)

# TODO: train the decision tree classifier on entire dataset
tree_full = DecisionTreeClassifier(criterion="entropy", max_depth=3)
tree_full = tree_full.fit(X, Y)

X_visualize = []; Y_visualize = []
time_0 = 0; time_1 = 0; time_2 = 0; time_3 = 0

for i,window_with_timestamp_and_label in slidingWindow(visualize_data, window_size, step_size):
    window_visualize = window_with_timestamp_and_label[:,1:-1]
    feature_names, x = extract_features(window_visualize)
    X_visualize.append(x)
    Y_visualize.append(window_with_timestamp_and_label[10, -1])

pred_vals_final = tree_full.predict(X_visualize)
count_0 = np.count_nonzero(pred_vals_final == 0)
count_1 = np.count_nonzero(pred_vals_final == 1)
count_2 = np.count_nonzero(pred_vals_final == 2)
count_3 = np.count_nonzero(pred_vals_final == 3)

#Added here
for i in range(len(pred_vals_final)-1):
    if pred_vals_final[i] == 0:
        time_0 += X_visualize[i+1][0] - X_visualize[i][0]
    elif pred_vals_final[i] == 1:
        time_1 += X_visualize[i][1] - X_visualize[i+1][1]
    elif pred_vals_final[i] == 2:
        time_2 += X_visualize[i+1][2] - X_visualize[i][2]
    elif pred_vals_final[i] == 3:
        time_3 += X_visualize[i+1][3] - X_visualize[i][3]

labels = ['Walking', 'Running', 'Sitting', 'Standing']
sizes = [count_0, count_1, count_2, count_3]
explode = (0, 0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

plt.bar(labels,sizes,color='blue',label='Activities v/s time')
plt.xlabel('Activities')
plt.ylabel('Minutes recorded')
plt.title('Activities v/s time')
plt.show()

def onActivityDetected(activity):
    """
    Notifies the user of the current activity
    """
    print("Detected activity:" + activity)

def predict(window):
    """
    Given a window of accelerometer data, predict the activity label.
    """
    # TODO: extract features over the window of data
    feature_names, feature_vector = extract_features(window)
    feature_vector = feature_vector.reshape(-1, 1)
    #print(feature_vector)

    # TODO: use classifier.predict(feature_vector) to predict the class label.
    # Make sure your feature vector is passed in the expected format
    pred_label = tree.predict(feature_vector.T)

    # TODO: get the name of your predicted activity from 'class_names' using the returned label.
    # pass the activity name to onActivityDetected()
    if(pred_label == 0):
        onActivityDetected("Walking")
    elif(pred_label == 1):
        onActivityDetected("Running")
    elif(pred_label == 2):
        onActivityDetected("Sitting")
    elif(pred_label == 3):
        onActivityDetected("Standing")
    time.sleep(0.2)
    print()
    return



count = 0
step_count = 0
buffer = []
stepindices = 0

def detectSteps(t,x_in,y_in,z_in):
    global count
    global step_count
    global buffer
    global mag_vals
    global stepindices
    global start_time
    """
    Accelerometer-based step detection algorithm.

    In this assignment, you will implement your step detection algorithm.
    Remember to use the global keyword if you would like to access global
    variables such as counters or buffers.
    """

    # TODO: Step detection algorithm
    mag = (x_in**2 + y_in**2 + z_in**2)**0.5


    if(count<10):
        buffer.append(mag)
        count+=1
    else:
        #print(buffer)
        #Detecting a step
        #prev_max_signal = buffer[0]
        no_steps = 0
        max_val = 0
        for i in buffer:
            if(i>3):
                no_steps+=1
                if(i>max_val):
                    max_val=i
        #stepindices = list(stepindices)
        stepindices = max_val

        if(no_steps>0):
            step_count+=2
            #mag_vals = buffer

            print("Number of steps detected =",step_count)
            print()

            #Calculating the calories
            stride_length = 0.57
            dist = step_count * stride_length
            time_elapsed = time.time() - start_time

            speed = dist/time_elapsed

            calories = 3 * speed * 4.5 #1.25*3600/1000
            print("Total calories burnt =", calories)

        buffer = []
        count = 0


    return


#################   Server Connection Code  ####################
#All predicting to be done here

start_time = time.time()

def recv_data():
    index=0
    global receive_socket
    global t, x, y, z
    global tvals, xvals, yvals, zvals, magvals

    test_data = []
    count = 0



    while True:
        message = connection.recv(1024)
        data = message.decode().split(",")
        if "accelerometer" in data[2]:
            print("Omit header line")
        else:
            try:
                t = int(float(data[2])*1000)
                x, y, z = [float(data[i])*10 for i in (3, 4, 5)]
            except ValueError:
                t = 2000
                x = 0.0
                y = 0.0
                z = 0.0
                continue
#            print("t=" + str(t) + "\t x=" + str(x) + "\t y=" + str(y) + "\t z=" + str(z))
            #print(x,y,z)

            # sensor_data.append(reorient(x,y,z))
            # index+=1
            #
            # while len(sensor_data) > window_size:
            #     sensor_data.pop(0)
            #
            # if (index >= step_size and len(sensor_data) == window_size):
            #     t = threading.Thread(target=predict, args=([np.asarray(sensor_data[:])],))
            #     t.start()
            #     index=0

            xvals = shift(xvals, 1, cval=0)
            xvals[0] = x

            yvals = shift(yvals, 1, cval=0)
            yvals[0] = y

            zvals = shift(zvals, 1, cval=0)
            zvals[0] = z

            test_data.append([x,y,z])
            if(count%25==0):
                predict(test_data)
                test_data=[]

            detectSteps(t,x,y,z)

# Helper function that animates the canvas
def animate(i):
    global tvals, xvals, yvals, zvals, magvals, stepindices

    step_marker_locs = np.nonzero(stepindices)
    step_marker_locs = list(step_marker_locs[0])

    try:
        ax1.clear()
        ax2.clear()
        ax1.plot(tvals, xvals, label="X")
        ax1.plot(tvals, yvals, label="Y")
        ax1.plot(tvals, zvals, label="Z")
        ax1.set_title('Real Time Acceleration')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Acceleration (m/s^2)')
        ax1.set_ylim(-40,40)

        if(len(step_marker_locs)>0):
            ax2.plot(tvals, magvals, '-gD', markevery=step_marker_locs,markersize=20)
        else:
            ax2.plot(tvals, magvals, '-g')
        ax2.set_title('Real Time Magnitude')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Acceleration (m/s^2)')
        ax2.set_ylim(0,40)
    except KeyboardInterrupt:
        quit()

try:
    #This socket is used to receive data from the data collection server
    receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receive_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #print(socket.gethostname())
    # receive_socket.bind((socket.gethostname(), 9800))
    receive_socket.bind((socket.gethostbyname(socket.gethostname()), 9800))

    receive_socket.listen(4) # become a server socket, maximum 5 connections
    connection, address = receive_socket.accept()

    t = 0
    x = 0
    y = 0
    z = 0

    #numpy array buffers used for visualization
    tvals = np.linspace(0,10,num=250)
    xvals = np.zeros(250)
    yvals = np.zeros(250)
    zvals = np.zeros(250)
    magvals = np.zeros(250)
    stepindices = np.zeros(250,dtype='int')

    socketThread = threading.Thread(target=recv_data, args=())
    socketThread.daemon = True # die when main thread dies
    socketThread.start()

    #Setup the matplotlib plotting canvas
    style.use('fivethirtyeight')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # Point to the animation function above, show the plot canvas
    ani = animation.FuncAnimation(fig, animate, interval=20)
    plt.show()

except KeyboardInterrupt:
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Quitting...")
    plt.close("all")
    quit()
#finally:
#    print("closing socket for receiving data")
#    receive_socket.shutdown(socket.SHUT_RDWR)
#    receive_socket.close()
