import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import argparse

from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelBinarizer

#Import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

#Function to give transparency to a color
def to_transparent(rgb, bg_rgb=[1, 1, 1], alpha=0.5):
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]

#Creates a Scatter plot with the different classes
def draw_dataset(X, y, stack_plot=False, plot_n=1, background=False, fig=None, ax=None):
	df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
	np.random.seed(40)
	if background:
		colors_basic = {0:to_transparent((1,0,0)), 1:to_transparent((0,0,1)), 2:to_transparent((0,1,0))}
		extra_colors = {key: value for (key, value) in [[i+3,to_transparent(np.random.rand(3))] for i in range(10)]}
		legend=False
		s=1
	else:
		colors_basic = {0:(1,0,0), 1:(0,0,1), 2:(0,1,0)}
		extra_colors = {key: value for (key, value) in [[i+3,np.random.rand(3)] for i in range(10)]}
		legend=True
		s=None
	colors = dict(colors_basic.items() + extra_colors.items())
	if plot_n==1:
		fig, ax = plt.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
		group.plot(ax=ax, kind='scatter', x='x', y='y', s=s, label=key, color=colors[key], legend=legend)
	if not stack_plot:
		plt.show()
	return fig, ax

#generates a dataset, either moons of blobs
def generate_dataset(n_samples=500, n_classes=2, type='moons'):
	if type=="moons":
		if n_classes!=2:
			print("[INFO] Dataset type moons only accepts 2 classes, setting number of classes to 2...")
		X, y = make_moons(n_samples=n_samples, noise=0.13)
	elif type=="blobs":
		X, y = make_blobs(n_samples=n_samples, n_features=2, centers=n_classes ,cluster_std=1.5, random_state=3)
	else:
		print("[INFO] Specified dataset hasn't been implemented")
	return X,y

#Create model
def create_model(classes=2, layers=[2, 2], lr=0.01):
    model = Sequential()
    input_shape = (2,)
    opt = SGD(lr=0.01)
    for i, layer in enumerate(layers):
        model.add(Dense(layer, input_shape=input_shape, name="Dense_layer_"+str(i+1)))
        model.add(Activation("relu", name="Activation_ReLu_"+str(i+1)))

    if classes==2:
        model.add(Dense(1, name="Output_layer"))
        model.add(Activation("sigmoid", name="Activation_Sigmoid"))
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(classes, name="Output_layer"))
        model.add(Activation("softmax", name="Activation_Softmax"))
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, show_shapes=True, to_file= 'model.png')
    #print(model.summary())
    return model

def train_model(X, y, n_classes=2, layers=[2,2], epochs=100, lr=0.01):
	start = time.time()
	if n_classes>2:
		lb = LabelBinarizer()
		y = lb.fit_transform(y)
	model = create_model(classes=n_classes, layers=layers, lr=0.01)
	model.fit(X, y, batch_size=32, epochs=epochs, verbose=0)
	end = time.time()
	print("\n[INFO] Training time {:.2f}s, if it took too long consider reducing model's size or dataset's size".format(end-start))
	return model

def Draw_decision_boundary(model, X, y, n_classes=2, res=300):
	start = time.time()
	#Create area to test with neural network
	max_value_x=X[:,0].max()*1.25
	max_value_y=X[:,1].max()*1.25
	min_value_x=X[:,0].min()*1.25
	min_value_y=X[:,1].min()*1.25
	step_size = res
	steps_x=(max_value_x-min_value_x)/step_size
	steps_y=(max_value_y-min_value_y)/step_size
	space1D_x = [min_value_x+f*steps_x for f in range(step_size+1)]
	space1D_y = [min_value_y+f*steps_y for f in range(step_size+1)]
	space2D = [[space1D_x[i], space1D_y[j]] for i in range(len(space1D_x)) for j in range(len(space1D_y))]
	space2D = np.asarray(space2D)
	#test the whole area on the neural network
	space2D_label = model.predict(space2D, batch_size=32)
	if n_classes==2:
		space2D_label = [1 if sp>0.5 else 0 for sp in space2D_label]
	else:
		space2D_label = space2D_label.argmax(axis=1)
	fig, ax = draw_dataset(space2D, space2D_label, stack_plot=True, background=True)
	end = time.time()
	print("\n[INFO] Drawing time {:.2f}s, if it took too long consider reducing resolution\n".format(end-start))
	draw_dataset(X, y, plot_n=2, fig=fig, ax=ax)

def main():
	#Default arguments
	type_ = "moons"
	layers = [20 for i in range(2)]
	epochs = 50
	n_classes = 2
	lr = 0.01

	#Parse arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=False, default=type_, type=str, help="Dataset type")
	ap.add_argument("-c", "--classes", required=False, default=n_classes, type=int, help="Number of classes")
	ap.add_argument("-l", "--layers", required=False, default=layers, type=int, nargs="+", help="path to the output model")
	ap.add_argument("-e", "--epochs", required=False, default=epochs, type=int, help="Number of epochs")
	ap.add_argument("-lr", "--learning_rate", required=False, default=lr, type=float, help="Learning Rate")
	args=vars(ap.parse_args())
	type_ = args["dataset"]
	layers = args["layers"]
	epochs = args["epochs"]
	n_classes = args["classes"]
	if type_ == "moons":
		n_classes = 2
		print("[INFO] Dataset moons only accepts 2 classes")
	lr = args["learning_rate"]

	#Create dataset
	X,y = generate_dataset(n_classes=n_classes, type=type_)

	#create and train model
	model = train_model(X, y, n_classes=n_classes, layers=layers, epochs=epochs, lr=lr)
	Draw_decision_boundary(model, X, y, n_classes=n_classes, res=300)

if __name__ == "__main__":
	main()
