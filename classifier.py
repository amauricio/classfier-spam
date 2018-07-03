import string
import random as rd
import numpy as np
import pandas as pd
import sys
from fastText import train_supervised 


data = pd.read_csv('spam.csv',encoding='ISO-8859-1');


stopwords = [ x.replace('\n', '') for x in open('stopwords.txt').readlines() ]
valdata = data.values

n_data = []
for x in valdata:
	value = x[1]
	## Eliminar stopwords
	#for stop in stopwords:
		#value = value.replace(' '+stop+' ', ' ')
	n_data.append([ x[0], value ])


n_data = pd.DataFrame(n_data)

count_train=4000

train_data = n_data.iloc[0:count_train,]
test_data = n_data.iloc[count_train:len(data),]

cot_file = 'data.train'
test_file = 'data.test'



def add_label(filename, label, text):
	with open(filename, 'a') as fm:
		fm.write('__label__'+label + ' ' + text + '\n')
		fm.flush()
		fm.close()

def apply_filter(td, filename):

	with open(filename, 'w') as sp:
		sp.write('')
	#in content
	ds = {}
	for f in td.values:
		target = f[0]
		text = f[1]
		if target not in ds:
			ds[target] = []
		ds[target].append(text)

	#append spam
	for value in ds['spam']:
		add_label(filename ,'spam', value)

	#append ham
	for value in ds['ham']:
		add_label(filename ,'ham', value)

apply_filter(train_data, cot_file)
apply_filter(test_data, test_file)

predict_ok = 0
predict_fail = 0


#Creamos el modelo
model = train_supervised(input=cot_file, epoch=50, wordNgrams=5, verbose=2, minCount=1, loss="hs")

##Testear el archivo test sin categoria
for d in test_data.values:
	pred_label, estimation = model.predict(d[1])
	predicted = pred_label[0].replace('__label__', '')
	real = d[0]
	if str(real) == str(predicted):
		predict_ok+=1
	else:
		predict_fail+=1


print('\nResults:')
ln = len(test_data)
print('Predicted True', predict_ok/ln)
print('Predicted False',predict_fail/ln)
print('\n')

nex, precision, recall = model.test(test_file)
print('N Examples', nex)
print('Precision', precision)
print('Recall', recall)

print('\n')
