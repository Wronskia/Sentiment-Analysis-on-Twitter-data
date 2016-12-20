import os
import subprocess
import numpy as np


#epochs=[2,5,12,15,20,30,40,50,70]
epochs=np.array(list(range(50)))+1
#lrs=[0.0005,0.001,0.05, 0.1, 0.25, 0.5,1.0]
#sizewords=[10,20,50,100,200,300,400]
sizeword=200
lr=0.001
#wordngram=2
gram=2
valid=[]
test=[]
for epoch in epochs:
    x=subprocess.Popen(["./fasttext supervised -input fastText_training.txt -output model -lr "+str(lr)+" -dim "+ str(sizeword)+" -wordNgrams "+ str(gram)+" -epoch "+str(epoch)], stdout=subprocess.PIPE, shell=True)
    (out2, err2)=x.communicate()
    proc = subprocess.Popen(["./fasttext test model.bin fastText_training.txt"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    valid.append(out.decode('utf8'))
    proc = subprocess.Popen(["./fasttext test model.bin fastText_validation.txt"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    test.append(out.decode('utf8'))
i=0

for epoch in epochs:
    #print(str(epoch)+" "+str(lr)+" "+str(sizeword)+" "+str(gram)+" "+str(valid[i][5:10])+" "+str(valid[i][16:21])+" "+str(test[i][5:10])+" "+str(test[i][16:21]))
    print("Model parameters : "+" "+str(epoch)+" "+str(lr)+" "+str(sizeword)+" "+str(gram)+" Training Accuracy "+str(valid[i][5:10])+" "+" Validation Accuracy "+str(test[i][5:10])+" ")

    i+=1
