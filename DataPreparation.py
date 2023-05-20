# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:08:39 2023

@author: Michael
"""
import pandas as pd

tableA = pd.read_csv(r'Pfadname\\Ordner\\tableA.csv')
tableB = pd.read_csv(r'Pfadname\\Ordner\\tableB.csv')

#Methode um die Datensätze einzulesen, aufzubereiten und zusammenzuführen
def gen(name):
    test = pd.read_csv(r'Pfadname\\Ordner\\'+name+'.csv')
    merged = pd.DataFrame(data={'label': [], 'left_Attribute': [], 'left_Attribute2': [], 'left_Attribute3': [], 
                                'right_Attribute': [], 'right_Attribute2': [], 'right_Attribute3': []})
    merged.index.name='id'
    for i in test.itertuples():
        row = pd.DataFrame(data={'label': [i.label],  'left_Attribute1': [tableA.at[i.ltable_id, "Attribute1"]],
                                 'left_Attribute2': [tableA.at[i.ltable_id, "Attribute2"]],
                                 'left_Attribute3': [tableA.at[i.ltable_id, "Attribute3"]],
                                 'right_Attribute1': [tableB.at[i.rtable_id, "Attribute1"]],
                                 'right_Attribute2': [tableB.at[i.rtable_id, "Attribute2"]], 
                                 'right_Attribute3': [tableB.at[i.rtable_id, "Attribute3"]]})
        row.index.name='id'
        row = row.astype({'label':'int'})
        merged = merged.append(row, ignore_index=True)
    merged.index.name='id'
    merged = merged.astype({'label':'int'})
    merged.to_csv(r'Pfadname\\Ordner\\'+name+'_merged.csv')

#Generierung eines Test-, Validations- und Trainingsdatensatzes
gen('test')
gen('valid')
gen('train')

#Deepmatcher-Verfahren
import deepmatcher as dm
train, validation, test = dm.data.process(path='Pfadname\\Ordner',train='train_merged.csv', validation='valid_merged.csv', test='test_merged.csv')

model = dm.MatchingModel(attr_summarizer='hybrid')

model.run_train(train, validation, best_save_path='best_model.pth', epochs=10, batch_size=32)

model.run_eval(test)

test_predictions = model.run_prediction(test, output_attributes=True)
test_predictions.head()
test_predictions.to_csv('Pfadname\\Ordner\\test_predictions.csv')