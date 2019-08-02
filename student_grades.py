import pandas as pd
import os
import sys
from math import sqrt
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cwd = os.path.abspath(__file__+"/..")

def create_dataset():
	df = pd.read_csv(cwd+"/data/student-mat.csv", sep=";")
	#list of columns to remove
	to_drop = [c for c in list(df.columns) if c not in ["sex", "age", "address", "Mjob", "guardian", "studytime", "higher", "internet", "absences", "G1", "G2", "G3"]]
	#remove columns
	df = df.drop(columns=to_drop, inplace=False).sample(frac=1).reset_index(drop=True)
	#change and rename data to appropriate inputs
	df["sex"] = df["sex"].replace({"M":1, "F":0})
	
	df["address"] = df["address"].replace({"U":1, "R":0})
	
	df["higher"] = df["higher"].replace({"yes":1, "no":0})
	
	df["internet"] = df["internet"].replace({"yes":1, "no":0})
	
	df.loc[df.Mjob != "at_home", "Mjob"] = 1
	
	df.loc[df.Mjob == "at_home", "Mjob"] = 0
	
	df = df.loc[df.guardian != "other"]
	
	df["guardian"] = df["guardian"].replace({"father":1, "mother":0})
	
	old_cols = ["sex", "age", "address", "Mjob", "guardian", "studytime", "higher", "internet", "absences", "G1", "G2", "G3"]
	new_cols = ["male", "age", "urban_address", "mother_works", "father_is_guardian", "studytime", "higher", "internet", "absences", "first_per_grade", "second_per_grade", "final_grade"]
	
	col_dict = dict(zip(old_cols, new_cols))
	
	df = df.rename(columns = col_dict)
	
	df.to_csv(cwd+"/data/student_dataset.csv", sep="\t")

def load_data():
	df = pd.read_csv(cwd+"/data/student_dataset.csv", sep="\t")
	return df

def create_model():
	model=Sequential()
	model.add(Dense(33, input_dim=11, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
	return model

def predict_grades():
	model = load_model('./models/student_grade.h5')	
	df = pd.read_csv(cwd+"/data/want_to_predict_grades.csv", sep="\t", header=None)
	#turn dataframe to list of inout arrays
	dataset = [np.array([np.asarray(x)]) for x in df.values.tolist()]
	output = []
	#use model to predict grade for each student
	for arr in dataset:
		output.append(model.predict(arr))
	print(output)

def main():
	#clean and extract daya we want from the downloaded dataset
	create_dataset()
	#load cleaned dataset
	df = load_data()
	#split data into training and testing values then get respective outputs and inputs
	dataset = df.values
	dataset = dataset[:360]
	train = dataset[:300]
	test = dataset[300:]

	train_inputs = train[:, 0:11]
	train_outputs = train[:,11]
	test_inputs = test[:, 0:11]
	test_outputs = test[:,11]

	#create Keras model to use
	model = create_model()
	#train the model using training data
	model.fit(train_inputs, train_outputs, epochs=10, batch_size=50)
	#evaluate model using testing data
	loss, mse = model.evaluate(test_inputs, test_outputs, batch_size=1, verbose=1)
	print("RMSE : ", mse)
	#save model so we dont have to train it again
	model.save('./models/student_grade.h5')
	#remove model from program memory
	del model
	#load and use trained model
	predict_grades()

if __name__ == "__main__":
	main()