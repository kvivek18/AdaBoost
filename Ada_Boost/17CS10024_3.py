#Roll Number   :- 17CS10024
#Name		   :- K.V.R.K.Vivek
#Assignment No :-3

import numpy as np
import pandas as pd
import math
from pprint import pprint

# find entropy for current data without any split
def entropy(Y):  

    Ent = 0
    values = Y.unique()
    for val in values:
    	temp=Y.value_counts()[val]
    	length=len(Y)
    	frac=temp/length	
    	frac=frac*math.log(frac)
    	Ent=Ent-frac
    return Ent



# calculate entropy gain on splitting with att attribute
def gain(X, Y, att):
	
	Ent_X = entropy(Y)
	values = X[att].unique()
	Ent_sum = 0
	for val in values:
		index = X.index[X[att] == val].tolist()
		Y_temp = Y.iloc[index]
		Y_temp = Y_temp.reset_index(drop=True)
		frac=len(Y_temp)
		length=len(Y)
		frac=frac/length
		frac=frac*entropy(Y_temp)
		Ent_sum = Ent_sum + frac
	pres_res=Ent_X-Ent_sum
	return pres_res



# Decide on which attribute to split based on Highest gain
def decide_att(X, Y):

	hi=X.keys()
	attribute = hi[0]
	_gain = 0
	for att in X.keys():
		temp = gain(X, Y, att)
		if temp<=_gain:
			continue
		else:
			_gain = temp
			attribute = att
	return attribute



# Get the sub data after splitting wit att attribute
def get_sub_data(X, Y, att, val):

	index = X.index[X[att] == val].tolist()
	X_temp = X.iloc[index, : ]
	Y_temp = Y.iloc[index]
	X_temp = X_temp.reset_index(drop=True)
	Y_temp = Y_temp.reset_index(drop=True)
	return X_temp, Y_temp



# Get the decision Tree based on Information gain, withou any pruning at any level
def get_tree(X, Y, count, p_att, tree=None):
	att = decide_att(X, Y)
	values = X[att].unique()
	if tree is None:                    
		tree = {}
		tree[att] = {}
	
	for val in values:
		X_sub, Y_sub = get_sub_data(X, Y, att, val)
		y_values = Y_sub.unique()
		for lol in range(0,1):
			yes = 0
			no = 0
		for y_val in y_values:
			if y_val != 'no':
				yes = Y_sub.value_counts()['yes']
			if y_val != 'yes':
				no = Y_sub.value_counts()['no']
		
		if no != 0:
			ratio = float(yes/no)
		else :
			ratio = 150

		if att == p_att:
			if yes <= no:
				return 'no'
			else:
				return 'yes'

		elif count > 1: # As only 3 attributes in this case
			if yes > no:
				tree[att][val] = 'yes'
			else:
				tree[att][val] = 'no'
		
		else :
			temp=count+1
			tree[att][val] = get_tree(X_sub, Y_sub, temp, att)
		
	return tree


# Function to detect no of correct and incorrect predictions
def test_accuracy(X_test, Y_test, tree,prob):
	correct = 0
	incorrect = 0
	epsilon=0
	rorw=[]
	for index, row, in X_test.iterrows():
	    ini = tree.keys()
	    pres_prob=prob[index] 
	    check=list(ini)
	    i = check[0]
	    val1 = row[i]
	    check=tree[i]
	    sec = check.keys()

	    for j in sec:
	    	if val1!=j:
	    		continue
	    	
	    	else:
	    		
	    		thr = tree[i][j]
	    		if (thr == 'yes') | (thr == 'no'):
	    			if Y_test[index]!=thr:
	    				epsilon=epsilon+pres_prob
	    				incorrect += 1
	    				rorw.append(0)
	    				continue

	    			else:
	    				correct += 1
	    				rorw.append(1)
	    				continue
	    			
	    			
	    		thr = thr.keys()
	    		temp=list(thr)
	    		k = temp[0]

	    		val2 = row[k]

	    		fou = tree[i][j][k].keys()
	    		for l in fou:
	    			if val2!=l:
	    				continue
	    			else:
	    				fiv = tree[i][j][k][l]
	    				if (fiv == 'yes') | (fiv == 'no'):
	    					if Y_test[index] != fiv:
	    						epsilon=epsilon+pres_prob
	    						incorrect += 1
	    						rorw.append(0)
	    						continue
	    					
	    					else:
	    						correct += 1
	    						rorw.append(1)
	    						continue

	    					
	    				fiv = fiv.keys()
	    				temp=list(fiv)
	    				m = temp[0]

	    				val3 = row[m]
	    				
	    				six = tree[i][j][k][l][m].keys()
	    				for n in six:
	    					if val3!=n:
	    						continue

	    					else:
		    					sev = tree[i][j][k][l][m][n]
		    					if Y_test[index] != sev:
		    						epsilon=epsilon+pres_prob
		    						incorrect += 1
		    						rorw.append(0)
		    						continue
		    						
		    					else:
		    						correct += 1
		    						rorw.append(1)
		    						continue
	return correct,incorrect,epsilon,rorw
             

def final_test(X_test,Y_test,tree_comp,alpha_values):
	correct=0
	incorrect=0

	for index, row, in X_test.iterrows():
		pres_power=0
		yes_count=[]
		for pres_tree in range(0,3):
			tree=tree_comp[pres_tree]
			ini = tree.keys()
			i=list(ini)[0]
			val1=row[i]
			sec=tree[i].keys()
			for j in sec:
				if val1==j:
					thr=tree[i][j]
					if(thr=='yes'):
						yes_count.append(1)
						continue
					if(thr=='no'):
						yes_count.append(0)
						continue
					thr=thr.keys()
					k=list(thr)[0]
					val2=row[k]
					fou=tree[i][j][k].keys()

					for l in fou:
						if val2==l:
							fiv=tree[i][j][k][l]
							if(fiv=='yes'):	
								yes_count.append(1)
								continue

							if(fiv=='no'):
								yes_count.append(0)
								continue

							fiv=fiv.keys()
							
							m=list(fiv)[0]

							val3=row[m]

							six=tree[i][j][k][l][m].keys()
							for n in six:
								if val3==n:
									sev=tree[i][j][k][l][m][n]
									if (sev=='yes'):
										yes_count.append(1)
										continue
									if(sev=='no'):
										yes_count.append(0)
										continue


			    				
		for pres_tree in range(0,3):
			pres_power=(yes_count[pres_tree])*alpha_values[pres_tree]+pres_power

		if(Y_test[index]=='yes'):
			if(pres_power>=0.5):
				correct=correct+1
		if(Y_test[index]=='yes'):
			if(pres_power<0.5):
				incorrect=incorrect+1
		if(Y_test[index]=='no'):
			if(pres_power>=0.5):
				incorrect=incorrect+1
		if(Y_test[index]=='no'):
			if(pres_power<0.5):
				correct=correct+1

	#print(correct,incorrect)
	return correct*100/(len(X_test))


	

X_data = pd.read_csv('data3_19.csv')

X_train = X_data.iloc[:,:-1].reset_index(drop=True)
Y_train = X_data.iloc[:,-1].reset_index(drop=True)

len_test=len(X_train)

prob=[]
for i in range(0,len_test):
	val=1/len_test
	prob.append(val)

indices=[]
for i in range(0,len_test):
	indices.append(i)

epsilon_values=[]
alpha_values=[]
tree_comp=[]

ind_selected=np.random.choice(indices,len_test,prob)
X_data_selected=X_train.iloc[ind_selected,:].reset_index(drop=True)
Y_data_selected=Y_train.iloc[ind_selected].reset_index(drop=True)

for viv in range(0,3):

	tree = get_tree(X_data_selected, Y_data_selected, 0, None)
	tree_comp.append(tree)

	correct,incorrect,epsilon,right_wrong= test_accuracy(X_data_selected, Y_data_selected, tree,prob)

	epsilon_values.append(epsilon)
	temp=epsilon*-1+1
	alpha=0.5*math.log(temp/epsilon)
	alpha_values.append(alpha)

	for i in range(0,len(right_wrong)):
		if right_wrong[i]==1:
			prob[i]=prob[i]*np.exp(alpha*-1)
		if right_wrong[i]==0:
			prob[i]=prob[i]*np.exp(alpha)


	tot_sum=sum(prob)
	for i in range(0,len(prob)):
		prob[i]=prob[i]/tot_sum

	ind_selected=np.random.choice(indices,len_test,prob)
	X_data_selected=X_data_selected.iloc[ind_selected,:].reset_index(drop=True)
	Y_data_selected=Y_data_selected.iloc[ind_selected].reset_index(drop=True)

	for i in range(0,len_test):
		prob[i]=1/len_test

temp=sum(alpha_values)

temp=pd.read_csv('test3_19.csv')

a=temp.columns[0]
b=temp.columns[1]
c=temp.columns[2]
d=temp.columns[3]

df_temp=pd.DataFrame({
	'pclass':[a],
	'age':[b],
	'gender':[c],
	'survived':[d]  
	})

X_test_data=temp.rename(columns={'1st':'pclass',
	'adult':'age',
	'male':'gender',
	'yes':'survived'})


X_test_data=X_test_data.append(df_temp,ignore_index=True,sort=True)

X_test=X_test_data.iloc[:,:-1].reset_index(drop=True)
Y_test=X_test_data.iloc[:,-1].reset_index(drop=True)

accuracy=final_test(X_test,Y_test,tree_comp,alpha_values)

print('The final accuracy of the ada boost algorithm is ',accuracy)
