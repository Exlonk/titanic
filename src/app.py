# Librerias
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import plotly.express as px
from sklearn import datasets
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import plotly.express as px
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
import math
from dash import dash_table,callback,Dash, html, dcc,  Input
from dash import Output, no_update, ctx
import plotly.express as px
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import data_science as ds
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import preprocessing
from dash_bootstrap_templates import load_figure_template # para los fondos de  las imagenes
import os        
import random                                    
from dash import DiskcacheManager
import diskcache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

discrete_color_graph = px.colors.diverging.BrBG

path_models= os.path.join(os.path.dirname(__file__),'models')
path_validation_curves = os.path.join(os.path.dirname(__file__),'validation_curves')
path_figures = os.path.join(os.path.dirname(__file__),'figures')
path_dataframes = os.path.join(os.path.dirname(__file__),'dataframes')

app = Dash(__name__,external_stylesheets=[dbc.themes.SUPERHERO],background_callback_manager=background_callback_manager,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, \
                             initial-scale=1.0'}]) # SOLAR, LUX

server = app.server

# <-------------------------------- FUNCTIONS -------------------------------> #

def load_plot_json(path,name):
    import plotly
    import json 
    import os
    with open(os.path.join(path,name+'.json'),'r+',encoding="utf-8") as f:
        read_ = f.read()
    figure = plotly.io.from_json(read_)
    return figure

# <----------------------------- DATA PREPARATION ---------------------------> #

# ----------------------------- Data Cleaning -------------------------------- #

# Pure Raw Dataframe 

df = pd.read_csv(os.path.join(os.path.dirname(__file__),'titanic.csv')) # './titanic.py'

columns_description=["Survived: Survival [0 no, 1 yes]", "Pclass = Ticket class","Name: Name",
                "Sex = Sex","Age = Age in years","Sibsp = Number of siblings or/and spouses aboard the Titanic",
                 "Parch = Number of parents or/and children  aboard the titanic",
                 "Ticket = Ticket number","Fare = Passanger fare",
                 "Cabin = Cabin number","Embarked = Port of embarkation"]         

table = ds.table(df,bgcolor="#0f2537",textcolor='#fff',bgheader='#07121B',columns_description=columns_description)  # [DASH COMPONENT TALBE]

shape = load_plot_json(path_figures,'shape')
shape.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
info_fig = load_plot_json(path_figures,'info_fig')  
info_fig.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

# Duplicates and Missing values

duplicates, missing_values = load_plot_json(path_figures,'duplicates'), load_plot_json(path_figures,'missing_values')   
duplicates.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
missing_values.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
missing_values.update_yaxes(range=(0,750))

# Number of categories

number_of_categories_1, number_of_categories_2 = load_plot_json(path_figures,'number_of_categories_1'), load_plot_json(path_figures,'number_of_categories_2')
number_of_categories_1.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
number_of_categories_2.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

# Class count

class_count=load_plot_json(path_figures,'class_count')
class_count.update_traces(showlegend=False)
class_count.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

# Outliers 

box_plot_pure = load_plot_json(path_figures,'box_plot_pure')
box_plot_pure.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

box_plot_ifo = load_plot_json(path_figures,'box_plot_ifo')
box_plot_lof = load_plot_json(path_figures,'box_plot_lof')
box_plot_ifo.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
box_plot_lof.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

box_plot_pure.update_xaxes(range=[-0.5,1.5])
box_plot_ifo.update_xaxes(range=[-0.5,1.5])
box_plot_lof.update_xaxes(range=[-0.5,1.5])

# X_transform_ifo = X_transform_ifo.drop(columns=['SibSp'])


# # <--------------------------------------------------------------------------> #
# # <----------------------------- DATA SPLIT ---------------------------------> #

X_train = pd.read_csv(os.path.join(path_dataframes,'X_train.csv'))
X_cv = pd.read_csv(os.path.join(path_dataframes,'X_cv.csv'))
X_test = pd.read_csv(os.path.join(path_dataframes,'X_test.csv'))
y_train = pd.read_csv(os.path.join(path_dataframes,'y_train.csv'))
y_cv = pd.read_csv(os.path.join(path_dataframes,'y_cv.csv'))
y_test = pd.read_csv(os.path.join(path_dataframes,'y_test.csv'))
X_train = pd.read_csv(os.path.join(path_dataframes,'X_train.csv'))
X_cv = pd.read_csv(os.path.join(path_dataframes,'X_cv.csv'))
X_test = pd.read_csv(os.path.join(path_dataframes,'X_test.csv'))
y_train = pd.read_csv(os.path.join(path_dataframes,'y_train.csv'))
y_cv = pd.read_csv(os.path.join(path_dataframes,'y_cv.csv'))
y_test = pd.read_csv(os.path.join(path_dataframes,'y_test.csv'))
# # Sample Distribution --------------------------------------------------------->

y_train_sample = pd.DataFrame(y_train).copy()
y_train_sample['set']='train'
y_cv_sample = pd.DataFrame(y_cv).copy()
y_cv_sample['set']='cv'
y_test_sample = pd.DataFrame(y_test).copy()
y_test_sample['set']='test'

sample_distribution_df = pd.concat([y_test_sample,y_train_sample,y_cv_sample],ignore_index=True)
sample_distribution_fig = px.scatter(sample_distribution_df,y='target',color='set',color_discrete_sequence=['#dfc27d','#543005',"#003c30"])
sample_distribution_fig.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

# <--------------------------------------------------------------------------> #
# <---------------------------- Feature Selection ---------------------------> #

# Numerical Features
mutual_classification_correlation_num = load_plot_json(path_figures,'mutual_classification_correlation_num')
anova_correlation = load_plot_json(path_figures,'anova_correlation')

# Categorical Features
chi2_correlation = load_plot_json(path_figures,'chi2_correlation')
mutual_classification_correlation_cat = load_plot_json(path_figures,'mutual_classification_correlation_cat')

# Layout graphs
chi2_correlation.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'},xaxis_title="")
mutual_classification_correlation_cat.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
mutual_classification_correlation_num.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
anova_correlation.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
chi2_correlation.update_yaxes(range=(0,185))
mutual_classification_correlation_cat.update_yaxes(range=(0,0.20))
anova_correlation.update_yaxes(range=(0,75))
mutual_classification_correlation_num.update_yaxes(range=(0,0.65))

# <--------------------------------------------------------------------------> #
# <-------------------------- MACHINE LEARNING MODELS -----------------------> #

if os.path.exists(os.path.join(os.path.dirname(__file__),'models')):
    path_models = os.path.join(os.path.dirname(__file__),'models')
else:
    os.mkdir(os.path.join(os.path.dirname(__file__),'models'))
    path_models= os.path.join(os.path.dirname(__file__),'models')

if os.path.exists(os.path.join(os.path.dirname(__file__),'validation_curves')):
    path_validation_curves = os.path.join(os.path.dirname(__file__),'validation_curves')
else:
    os.mkdir(os.path.join(os.path.dirname(__file__),'validation_curves'))
    path_validation_curves = os.path.join(os.path.dirname(__file__),'validation_curves')

# Polynomial Model
import os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
import joblib
from sklearn.preprocessing import PolynomialFeatures

# Degree optimization
print('Entering... Polynomial')

f1_train_cv_poly_graph, accuracy_train_cv_poly_graph, poly_parameters = \
     ds.single_polynomial_classification(LogisticRegression(),X_train,y_train.values.ravel(),X_cv,y_cv.values.ravel(),
       degree=10,path_=path_models,mode='design',save=True,
                              estimator_grid={'max_iter':[5000],'C':[i for i in range(1,60)],'class_weight':['balanced'],
                                                    'solver':['liblinear']},grid_par={'cv':10,'n_jobs':-1},color_line=['#dfc27d','#543005'])

f1_train_cv_poly_graph.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
accuracy_train_cv_poly_graph.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
# Main Model

poly_model = joblib.load(os.path.join(path_models,'LogisticRegression'+'_polynomial_classification_3.joblib'))
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_cv_poly = poly_features.fit_transform(X_cv)
X_test_poly = poly_features.fit_transform(X_test)
y_pred_train_prob_poly =  poly_model.predict_proba(X_train_poly)[:,1]
y_pred_cv_prob_poly =  poly_model.predict_proba(X_cv_poly)[:,1]
y_pred_test_prob_poly =  poly_model.predict_proba(X_test_poly)[:,1]
pr_train_poly, cm_train_poly,pr_cv_poly,cm_cv_poly,pr_test_poly,cm_test_poly,threshold_poly \
 = ds.binary_classification_model_evaluation(y_train,y_pred_train_prob_poly,\
     y_cv,y_pred_cv_prob_poly,y_test,y_pred_test_prob_poly,zero='No-Survived',one='Survived',color_line='yellow') 

sets_ = ['train','cv','test']

for i in sets_:
    vars()['pr_'+i+'_poly'].update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
    vars()['cm_'+i+'_poly'].update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

# Model Calibration

calibration_curve_poly, histogram_poly = ds.calibration_curve('polynomial ',y_true=y_test,y_pred=y_pred_test_prob_poly,
                                        color_line='#80cdc1',color_his='#944f69')
calibration_curve_poly.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
histogram_poly.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

# Ensemble Tree

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.calibration import CalibratedClassifierCV

tree_mode = 'design' # Work mode 'train' to train the model, 'design' to design the page layout with a trained model
tree_save_model = True # Save the model

print('Entering... Tree')

if tree_mode == 'train':
    parameters_tree = {
    "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
    "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight" : [ 1, 3, 5, 7 ],
    "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
    "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    }

    tree_classifier = XGBClassifier()
    tree_model=GridSearchCV(tree_classifier,param_grid=parameters_tree,\
                            scoring='roc_auc',n_jobs=-1)

    tree_model.fit(X_train,y_train)
    
    y_pred_train_prob_tree = tree_model.predict_proba(X_train)[:,1]
    y_pred_cv_prob_tree = tree_model.predict_proba(X_cv)[:,1]
    y_pred_test_prob_tree = tree_model.predict_proba(X_test)[:,1]

    if tree_save_model == True:
        joblib.dump(tree_model,os.path.join(path_models,'ensemble_tree.joblib'))

elif tree_mode == 'design':
    tree_model = joblib.load(os.path.join(path_models,'ensemble_tree.joblib'))
    y_pred_train_prob_tree = tree_model.predict_proba(X_train)[:,1]
    y_pred_cv_prob_tree = tree_model.predict_proba(X_cv)[:,1]
    y_pred_test_prob_tree = tree_model.predict_proba(X_test)[:,1]

# Model Evaluation

pr_train_tree, cm_train_tree,pr_cv_tree,cm_cv_tree,pr_test_tree,cm_test_tree,threshold_tree \
 = ds.binary_classification_model_evaluation(y_train,y_pred_train_prob_tree,\
     y_cv,y_pred_cv_prob_tree,y_test,y_pred_test_prob_tree,zero='No-Survived',one='Survived',color_line='yellow') 

sets_ = ['train','cv','test']

for i in sets_:
    vars()['pr_'+i+'_tree'].update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
    vars()['cm_'+i+'_tree'].update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

y_pred_train_tree = (y_pred_train_prob_tree >threshold_tree).astype(int)
y_pred_cv_tree= (y_pred_cv_prob_tree > threshold_tree).astype(int)
y_pred_test_tree= (y_pred_test_prob_tree >threshold_tree).astype(int)

# Model Calibration

calibration_curve_tree, histogram_tree = ds.calibration_curve('Ensemble Tree',y_true=y_test,y_pred=y_pred_test_prob_tree,color_line='#80cdc1',color_his='#944f69')
calibration_curve_tree.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
histogram_tree.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

# Neural Network -------------------------------------------------------------->

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from joblib import dump, load
from scikeras.wrappers import KerasClassifier
import pickle

neural_network_graph = ds.neural_network_fig([1,70,30,10,5,1])
neural_network_graph.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

neural_mode = 'design' # Work mode 'train' to train the model, 'design' to design the page layout with a trained model
neural_save_model = True # Save the model

print('Entering... Neural')

if neural_mode == 'train':

    neural_network = Sequential([
        Dense(units=500, activation = 'sigmoid'),#,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(units=300, activation = 'sigmoid'),#,kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(units=100, activation = 'sigmoid'),#,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(units=50, activation = 'sigmoid'),#,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(units=1, activation = 'linear')#,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    ])
    neural_network.build(input_shape=X_train.shape)
    neural_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss = BinaryCrossentropy(from_logits=True))

    neural_network.fit(X_train,y_train,epochs=200)
    y_pred_train_prob_network =  (tf.nn.sigmoid(neural_network.predict(X_train))).numpy().flatten() # neural_network.predict_proba(X_train)[:,1] #
    y_pred_cv_prob_network =  (tf.nn.sigmoid(neural_network.predict(X_cv))).numpy().flatten() # neural_network.predict_proba(X_cv)[:,1] # 
    y_pred_test_prob_network =  (tf.nn.sigmoid(neural_network.predict(X_test))).numpy().flatten() # neural_network.predict_proba(X_test)[:,1] # 

    # y_pred_train_prob_network =  tf.nn.sigmoid(neural_.predict(X_train)).numpy()
    # y_pred_cv_prob_network = tf.nn.sigmoid(neural_.predict(X_cv)).numpy()
    # y_pred_test_prob_network =   tf.nn.sigmoid(neural_.predict(X_test)).numpy()

    if neural_save_model == True:

        tf.keras.models.save_model(neural_network,os.path.join(path_models,'neural_network.h5'))

elif neural_mode == 'design':
    # file =open(os.path.join(path_models,'neural_network.pk'),'rb')
    # neural_network = pickle.load(file) 
    # neural_model = tf.keras.models.load_model(os.path.join(path_models,'neural_network.h5'))
    # neural_model= KerasClassifier(model=neural_model, epochs=100, batch_size=5, verbose=0,optimizer='adam',loss = BinaryCrossentropy())
    # neural_network = CalibratedClassifierCV(neural_model)
    # neural_network.fit(X_train,y_train)
    #neural_model = joblib.load(os.path.join(path_models,'neural_network.joblib'))
    neural_model = tf.keras.models.load_model(os.path.join(path_models,'neural_network.h5'))
    y_pred_train_prob_network = (tf.nn.sigmoid(neural_model.predict(X_train))).numpy().flatten() # (tf.nn.sigmoid(neural_model.predict(X_train))).numpy()
    y_pred_cv_prob_network = (tf.nn.sigmoid(neural_model.predict(X_cv))).numpy().flatten() # (tf.nn.sigmoid(neural_model.predict(X_cv))).numpy()
    y_pred_test_prob_network =  (tf.nn.sigmoid(neural_model.predict(X_test))).numpy().flatten() # (tf.nn.sigmoid(neural_model.predict(X_test))).numpy()
        
# Model Evaluation

pr_train_network, cm_train_network,pr_cv_network,cm_cv_network,pr_test_network,cm_test_network,threshold_network \
 = ds.binary_classification_model_evaluation(y_train,y_pred_train_prob_network,\
     y_cv,y_pred_cv_prob_network,y_test,y_pred_test_prob_network,zero='No-Survived',one='Survived',color_line='yellow') 

sets_ = ['train','cv','test']

for i in sets_:
    vars()['pr_'+i+'_network'].update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
    vars()['cm_'+i+'_network'].update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})


y_pred_train_network = (y_pred_train_prob_network >threshold_network).astype(int)
y_pred_cv_network= (y_pred_cv_prob_network > threshold_network).astype(int)
y_pred_test_network= (y_pred_test_prob_network >threshold_network).astype(int)

# Model Calibration

calibration_curve_network, histogram_network = ds.calibration_curve('Neural Network',y_true=y_test,y_pred=y_pred_test_prob_network
                                                     ,color_line='#80cdc1',color_his='#944f69')
calibration_curve_network.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
histogram_network.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

# ENSEMBLE CLASSIFIER ---------------------------------------------------------->

# from sklearn.ensemble import VotingClassifier

# ensemble_mode = 'train' # Work mode 'train' to train the model, 'design' to design the page layout with a trained model
# ensemble_save_model = True # Save the model

# print('Entering... Ensemble')

# if ensemble_mode == 'train':

#     ensemble_model = VotingClassifier(estimators=[('neural',neural_network),('tree', tree_model), ('poly', poly_model)],voting='soft', weights=[2,1,2])
#     ensemble_model.fit(X_train,y_train)
#     y_pred_train_prob_ensemble = ensemble_model.predict_proba(X_train)[:,1]
#     y_pred_cv_prob_ensemble = ensemble_model.predict_proba(X_cv)[:,1]
#     y_pred_test_prob_ensemble = ensemble_model.predict_proba(X_test)[:,1]
    
#     if ensemble_save_model == True:
#         joblib.dump(ensemble_model,os.path.join(path_models,'ensemble.joblib'))

# elif ensemble_mode == 'design':

#     ensemble_model = joblib.load(os.path.join(path_models,'ensemble.joblib'))
#     y_pred_train_prob_ensemble = ensemble_model.predict_proba(X_train)[:,1]
#     y_pred_cv_prob_ensemble = ensemble_model.predict_proba(X_cv)[:,1]
#     y_pred_test_prob_ensemble = ensemble_model.predict_proba(X_test)[:,1]

# # Figures and treshold

# pr_train_ensemble, cm_train_ensemble,pr_cv_ensemble,cm_cv_ensemble,pr_test_ensemble,cm_test_ensemble,threshold_ensemble \
#  = ds.binary_classification_model_evaluation(y_train,y_pred_train_prob_ensemble,\
#      y_cv,y_pred_cv_prob_ensemble,y_test,y_pred_test_prob_network,zero='No-Survived',one='Survived',color_line='yellow') 

# sets_ = ['train','cv','test']

# for i in sets_:
#     vars()['pr_'+i+'_ensemble'].update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
#     vars()['cm_'+i+'_ensemble'].update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

# y_pred_train_ensemble = (y_pred_train_prob_ensemble >threshold_network).astype(int)
# y_pred_cv_ensemble= (y_pred_cv_prob_ensemble > threshold_network).astype(int)
# y_pred_test_ensemble= (y_pred_test_prob_ensemble >threshold_network).astype(int)

# # Model Calibration

# calibration_curve_ensemble, histogram_ensemble = ds.calibration_curve('Ensemble',\
#     y_true=y_test,y_pred=y_pred_test_prob_ensemble,color_line='#80cdc1',color_his='#944f69')
# calibration_curve_ensemble.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
# histogram_ensemble.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})

# <---------------------------------- DEPLOY --------------------------------> #
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

df_ = pd.read_csv(os.path.join(os.path.dirname(__file__),'titanic.csv')) # './titanic.py'

# Quitar Filas Irrelevantes

df_.drop(columns=['Cabin','Ticket','Name'],inplace=True)
df_.rename(columns={'Survived':'target'},inplace=True)
X_raw = df_.drop(columns=['target']).copy()
y_raw = df_['target'].copy()
numerical_data = ['Age','Fare']
categorical_data = list(df_.columns[[i not in numerical_data and i != 'target' for i in df_.columns]])
encoding_data  = ['Sex','Embarked']

# Split, es necesario cambiar el nombre de la columna objetivo a 'target'

X_train, X_rem, y_train, y_rem = train_test_split(X_raw,y_raw, train_size=0.6,random_state=42)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_train.reset_index(inplace=True,drop=True)   
y_train.reset_index(inplace=True,drop=True)  

def name_columns(transformer_names,transformer):
    columns_name = []
    for i in transformer.get_feature_names_out():
        for k in transformer_names:
            if k in i:
                name = i.replace(k,'')
                columns_name.append(name)
    return columns_name

def transform_instance(preprocessor_imputer,preprocessor_encoder,outlier_,
num,cat,scaler,X_predict):
    numerical_data = ['Age','Fare']
    categorical_data = list(df_.columns[[i not in numerical_data and i != 'target' for i in df_.columns]])
    encoding_data  = ['Sex','Embarked']
    X_predict = preprocessor_imputer.transform(X_predict)
    columns_name = name_columns(['numerical__','categorical__','remainder__'],preprocessor_imputer)
    X_predict = pd.DataFrame(X_predict,columns=columns_name)
    X_predict = preprocessor_encoder.transform(X_predict)
    columns_name = name_columns(['encoder__','remainder__'],preprocessor_encoder)
    X_predict = pd.DataFrame(X_predict,columns=columns_name)
    categorical_data = [x for x in X_predict.columns if x not in numerical_data]

    df = X_predict
    X_bool_outliers = outlier_.predict(df[numerical_data])
    mask = X_bool_outliers != -1
    X_predict = df.iloc[mask,:]  

    X_predict = num.transform(X_predict)
    columns_name = name_columns(['numerical_feature__','remainder__'],num)
    X_predict = pd.DataFrame(X_predict,columns=columns_name)
    numerical_data = [x for x in X_predict.columns if x not in categorical_data]

    X_predict = cat.transform(X_predict)
    columns_name = name_columns(['categorical_feature__','remainder__'],cat)
    X_predict = pd.DataFrame(X_predict,columns=columns_name)

    categorical_data = [x for x in X_predict.columns if x not in numerical_data]
    numerical_data = [x for x in X_predict.columns if x not in categorical_data]

    X_predict = scaler.transform(X_predict)
    columns_name = name_columns(['scaler__','remainder__'],scaler)
    X_predict = pd.DataFrame(X_predict,columns=columns_name)

    categorical_data = [x for x in X_predict.columns if x not in numerical_data]
    numerical_data = [x for x in X_predict.columns if x not in categorical_data]

    X_predict.reset_index(inplace=True,drop=True) 
    return X_predict

# Preprocessing pipline

preprocessor_imputer = ColumnTransformer([
                ('numerical',SimpleImputer(strategy='median'),numerical_data),
                ('categorical',SimpleImputer(strategy='most_frequent'),categorical_data),
            ],verbose_feature_names_out=True,remainder='passthrough')
            
X_train = preprocessor_imputer.fit_transform(X_train)

columns_name = name_columns(['numerical__','categorical__','remainder__'],preprocessor_imputer)
X_train = pd.DataFrame(X_train,columns=columns_name)

preprocessor_encoder = ColumnTransformer([
        ('encoder',OneHotEncoder(),encoding_data)
    ],verbose_feature_names_out=True,remainder='passthrough')
    
X_train = preprocessor_encoder.fit_transform(X_train)

columns_name = name_columns(['encoder__','remainder__'],preprocessor_encoder)
X_train = pd.DataFrame(X_train,columns=columns_name)

categorical_data = [x for x in X_train.columns if x not in numerical_data]

# Remove Outliers

outlier_ = IsolationForest()
df = pd.concat([X_train,y_train],axis=1)
X_bool_outliers = outlier_.fit_predict(df[numerical_data])
mask = X_bool_outliers != -1
X_train, y_train = df.drop(columns=['target']).iloc[mask,:], pd.DataFrame(df['target'][mask])

# Feature selection

num = ColumnTransformer([('numerical_feature', SelectKBest(score_func=r_regression, k=1),numerical_data)],verbose_feature_names_out=True,remainder='passthrough')
X_train = num.fit_transform(X_train,y_train)

columns_name = name_columns(['numerical_feature__','remainder__'],num)
X_train = pd.DataFrame(X_train,columns=columns_name)

numerical_data = [x for x in X_train.columns if x not in categorical_data]

cat = ColumnTransformer([('categorical_feature', SelectKBest(score_func=chi2, k=8),categorical_data)],verbose_feature_names_out=True,remainder='passthrough')
X_train = cat.fit_transform(X_train,y_train)

columns_name = name_columns(['categorical_feature__','remainder__'],cat)
X_train = pd.DataFrame(X_train,columns=columns_name)

categorical_data = [x for x in X_train.columns if x not in numerical_data]
numerical_data = [x for x in X_train.columns if x not in categorical_data]

scaler = ColumnTransformer([('scaler',StandardScaler(),numerical_data)],verbose_feature_names_out=True,remainder='passthrough')
X_train = scaler.fit_transform(X_train,y_train)

columns_name = name_columns(['scaler__','remainder__'],scaler)
X_train = pd.DataFrame(X_train,columns=columns_name)

# Table 

columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

input_table = dash_table.DataTable(id='table-editing-simple',
        columns = [{"name": i, "id": i, "deletable": False, "selectable": False, "hideable": False} for i in columns], 
        data = [{'Pclass':3, 'Sex':'female', 'Age':25, 'SibSp':0, 'Parch':0, 'Fare':71, 'Embarked':'S'}],
        editable=True,              # allow editing of data inside all cells
        # "whiteSpace": "pre-line": permite dividir la información de las celdas en más de una columna
        style_data={                # overflow cells' content into multiple lines
            'whiteSpace': 'normal',
            'height': 'auto',
            'color': 'red',
            'backgroundColor': "#0f2537"
        },
        style_header={
            'backgroundColor': '#07121B',
             'color': '#fff',
          },
        )

# <-------------------------- Figure color layout ---------------------------> #

# all_variables = dir()
 
# for name in all_variables:
#     if type(vars()[name]) == type(shape):
#         vars()[name].update_layout(colorway = px.colors.diverging.BrBG)

# <----------------------------- Dash Layout --------------------------------> #

draw_figure_buttons = {'modeBarButtonsToAdd':['drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ]}

app.layout = dbc.Container([ 

    dbc.Row(dbc.Col([html.H1('Titanic DataFrame')],width=6,className="title")),
     dbc.Row(dbc.Col([html.H4('by Exlonk Gil')],width=12)),
    dbc.Row(dbc.Col([html.H2('Raw Data Insights')],width=12,    
    style={"padding-top":"1rem","padding-bottom":"1rem","textAlign":"center"})),

    dbc.Row(dbc.Col([html.P('This dataset shows the survivors to the titanic \
    accident (1: Survive, 0: Not Survive), because of this, it is a classifier \
    problem and due to its small size, the estimators used to modeling can be \
    very widely. For a detailed description of each feature, the table headers can be hovered.')])),

    dbc.Row(dbc.Col(html.Br(),width=12)), # LINEA EN BLANCO

    dbc.Row(dbc.Col([table],width=12)),

    dbc.Row(dbc.Col(html.Br(),width=12)), # LINEA EN BLANCO

    dbc.Row(dbc.Col([html.P('It can be seen that the features are mainly categorical, \
    and that there are some features that lack importance, such as the feature "name" \
    and the "ticket".')])),

    # Raw Data visualization
    dbc.Row(dbc.Col(html.Br(),width=12)), # Blank space

    dbc.Row(dbc.Col([html.P('The shape figure shows how many features and instances \
    are in the dataframe, the data type figure shows the type and number of each \
    feature, and the target categories histograms show the data imbalance.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=shape)],width=3),
             dbc.Col([dcc.Graph(figure=info_fig)],width=6),
             dbc.Col([dcc.Graph(figure=class_count)],width=3)]),

    dbc.Row(dbc.Col([html.P('It can be seen from these graphs that the data set \
    has few features and only three of them are numerical, besides the target data \
    is unbalanced.')])),

    # <------------------------- DATA CLEANING -------------------------> #

    dbc.Row(dbc.Col([html.H2('Data Cleaning')],width=12,
    className="title",style={"textAlign": "center"})),
    
    # Duplicates and missing values

    dbc.Row(dbc.Col([html.H3('Duplicates and Missing Values')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('This section shows if there is some duplicated\
                              rows in the dataframe ' 
                              ' and the number of missing data per feature')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=duplicates)],width=3),\
             dbc.Col([dcc.Graph(figure=missing_values)],width=9)]),

    dbc.Row(dbc.Col([html.P("The feature 'Cabin' has 690 missing values, this is \
        more than 75% of the data, on account of that this column is dropped, \
        whereas the feature 'Age' can be imputed using some characteristics like \
        its mean, median or an estimator like the Knnimputer.")])),


    # Number of categories

    dbc.Row(dbc.Col([html.H3('Categories Insights')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P("It is important to know what type of categorical \
    data is present in the data set, since with this information, it can be performed \
    a specific encoding. The data set has five categorical features: two nominal, \
    'sex' and 'embarked', and three ordinal, 'Pclass', 'SibSp' and 'Parch'.\
    Each one has its own unique category, and the next graphs illustrate the \
    type and amount of each one.")])),

    dbc.Row(dbc.Col([dcc.Graph(figure=number_of_categories_1)],width=12)),

    dbc.Row(dbc.Col([dcc.Graph(figure=number_of_categories_2)],width=12)),
    
    dbc.Row(dbc.Col([html.P("For each feature, the size of \
    the square indicates  how much data belongs to the category labeled. One Hot \
    Encoder is applied to the two nominal features, and no transformation is done \
    for the rest of the features.")])),

    # Outlier identification

    dbc.Row(dbc.Col([html.H3('Outlier Identification')],width=12,
                            className="subtitle", style={"textAlign": "left"})),
    
    dbc.Row(dbc.Col([html.P('Since this is a purely visual aid, a rescaling of \
                            all features is used, which rescaling (called Robust \
                            Scaler) is robust to outliers, this was done to take \
                            into account the target')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=box_plot_pure)],width=4),
        dbc.Col([dcc.Graph(figure=box_plot_lof)],width=4),
        dbc.Col([dcc.Graph(figure=box_plot_ifo)],width=4)]),

     # <------------------------- FEATURE SELECTION -------------------------> #

     dbc.Row(dbc.Col([html.H2('Feature Selection')],width=12,
                          className="title",style={"textAlign": "center"})),   

     dbc.Row(dbc.Col([html.P('This section shows some correlation metrics taking \
                               into account the nature of the predictors, \
                              that is, if they are categorical or numerical.')])),

    # # Pearson Correlation

    dbc.Row(dbc.Col([html.H3('Statistical Feature Selection')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.H5('Numerical Correlation')],width=12,
                             className="subtitle",style={"textAlign": "left"})),
    
    dbc.Row([ dbc.Col([dcc.Graph(figure=mutual_classification_correlation_num)],width=6),\
             dbc.Col([dcc.Graph(figure=anova_correlation)],width=6)]),

    dbc.Row(dbc.Col([html.P("The mutual information classification is a method that allows \
        to encounter relationship between the categorical target and a numerical or categorical \
        feature, the relationship measure between two random variables is a non-negative value, \
        which decides the dependency between the variables. It is equal to zero if and only if two \
        random variables are independent. When a variable is numeric and one is categorical, such \
        as numerical input variables and a classification target variable in a classification task, \
        an ANOVA f-test can be used. The anova and mutual information correlation shows that \
        'fare' is more related to the target than the age.")])),

    dbc.Row(dbc.Col([html.H5('Categorical Correlation')],width=12,
                             className="subtitle",style={"textAlign": "left"})),
    
    dbc.Row([dbc.Col([dcc.Graph(figure=chi2_correlation)],width=6),\
             dbc.Col([dcc.Graph(figure=mutual_classification_correlation_cat)],width=6)]),

     dbc.Row(dbc.Col([html.P("Pearson's chi-squared statistical hypothesis test is \
        a test for independence between categorical variables. Among the categorical features, the 'sex', 'class' and \
        'embarked port' are the most relevant features. The importance of 'Parch' is vague.")])),


    dbc.Row(dbc.Col([html.P('For the feature selection and data preprocessing many \
    combinations of transformations were made, the logistic regression model was \
    used as estimator (this because of the limited resources).')])),

    dbc.Row(dbc.Col([html.P('At the final stage, the pipeline selected for the \
        data preprocessing step is: ')])),

    dbc.Row(dbc.Col([html.P("OneHotEncoder for the categorical data ('sex' and 'Embarked'), \
     IsolationForest for the outlier data, SelectKBest(score_func=r_regression, k=1) for \
     the numerical data, keep all categorical features and StandardScaler to rescale the data.")])),

    # # <------------------------- MACHINE LEARNING MODELS -------------------------> #

    dbc.Row(dbc.Col([html.H2('Machine Learning Models')],width=12,
                         className="title",style={"textAlign": "center"})), 

    dbc.Row(dbc.Col([html.P('Different machine learning models were used to prove \
    this data set, its election was made by its diversity nature.\
    Due to the limited resources, for the preprocessing step different models were \
    proven against the logistic model.')])),

    dbc.Row(dbc.Col([html.H3('Sample Distribution')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('The figure shows the distributed labels of \
    the target on the training, validation and test set.')])),

    dbc.Row(dbc.Col([dcc.Graph(figure=sample_distribution_fig)],width=12)),
   
    # Polynomial Classification --------------------------------------------------->

    dbc.Row(dbc.Col([html.H3('Polynomial Classification [LogisticRegression]')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('The logistic regression is also known in the literature \
    as logit regression, maximum-entropy classification (MaxEnt) or the log-linear \
    classifier. It’s an extension of the linear regression model for classification \
    problems. In this case, multiple features were created by taking the n degree \
    of each feature in the original data set.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=f1_train_cv_poly_graph,config=draw_figure_buttons)],width=8),
             dbc.Col([dcc.Graph(figure=accuracy_train_cv_poly_graph)],width=4)]),

    dbc.Row(dbc.Col([html.P('The best degree with the lowest gap between the \
    train and CV accuracy is three. The next graphs are the confusion matrix \
    and the F1 metric. The first graph shows how much accuracy the model has \
    for each category, and the second one is used to encounter the best threshold \
    (the value used to label the probability as 0 or 1) using the highest value of F1.')])),

    # Model Evalutaion

    dbc.Row([dbc.Col([dcc.Graph(figure=cm_train_poly)],width=6),dbc.Col([dcc.Graph(figure=pr_train_poly)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=pr_cv_poly)],width=6),dbc.Col([dcc.Graph(figure=cm_cv_poly)],width=6)]),
   
    dbc.Row([dbc.Col([dcc.Graph(figure=cm_test_poly)],width=6),dbc.Col([dcc.Graph(figure=pr_test_poly)],width=6)]),

    dbc.Row(dbc.Col([html.P('In the training set, the model is more accurate than \
    79% on each label, its best threshold is nearly 0.5, and the average precision \
    is 0.86, so the model is doing well on this set.  In the training set and test \
    set, the accuracy is at least 69%.  In conclusion, a polynomial regression model \
    performs well in this scenario. Classifiers with a well calibrated output are \
    probabilistic, for which the output of the predict_proba method can be directly \
    interpreted as a confidence level. The following graphs show how well calibrated \
    is the model, A well-calibrated model must have a calibration curve that is \
    increasing nearly linear and a histogram with large counting of probabilities \
    near zero and one.')])),

    # Model Calibration

    dbc.Row([dbc.Col([dcc.Graph(figure=calibration_curve_poly)],width=8),dbc.Col([dcc.Graph(figure=histogram_poly)],width=4)]),


    dbc.Row(dbc.Col([html.P('In this case, the logistic model calibration graphs \
    appear to have some good performance. The histogram has large counting \
    before 0.3 and after 0.8 and the curve is increasing, although some erratics points.')])),

    dbc.Row(dbc.Col([html.H3('Ensemble Tree')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('For this model multiple hyperparameters were tune, \
    like the learning rate, the max depth, gamma etc.  Its accuracy is superior \
    to the polynomial one, as its calibration curve, in specific, the curve has \
    less erratic values. The histogram does not show a significant improvement.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=cm_train_tree)],width=6),dbc.Col([dcc.Graph(figure=pr_train_tree)],width=6)]),
    dbc.Row([dbc.Col([dcc.Graph(figure=pr_cv_tree)],width=6),dbc.Col([dcc.Graph(figure=cm_cv_tree)],width=6)]),
    dbc.Row([dbc.Col([dcc.Graph(figure=cm_test_tree)],width=6),dbc.Col([dcc.Graph(figure=pr_test_tree)],width=6)]),
    
    # Model Calibration

    dbc.Row([dbc.Col([dcc.Graph(figure=calibration_curve_tree)],width=8),dbc.Col([dcc.Graph(figure=histogram_tree)],width=4)]),

    # Neural Network --------------------------------------------------------> #
    
    dbc.Row(dbc.Col([html.H3('Neural Network')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('The next graph is a representation of the neural \
        network model used, it shows the number and relative size of the layers \
        as the relative number of connections')])),
   
    dbc.Row(dbc.Col([dcc.Graph(figure=neural_network_graph)],width=12)),
    
    # Evaluation

    dbc.Row([dbc.Col([dcc.Graph(figure=cm_train_network)],width=6),dbc.Col([dcc.Graph(figure=pr_train_network)],width=6)]),
    dbc.Row([dbc.Col([dcc.Graph(figure=pr_cv_network)],width=6),dbc.Col([dcc.Graph(figure=cm_cv_network)],width=6)]),
    dbc.Row([dbc.Col([dcc.Graph(figure=cm_test_network)],width=6),dbc.Col([dcc.Graph(figure=pr_test_network)],width=6)]),

    # Model Calibration

    dbc.Row([dbc.Col([dcc.Graph(figure=calibration_curve_network)],width=8),dbc.Col([dcc.Graph(figure=histogram_network)],width=4)]),

    dbc.Row(dbc.Col([html.P('Interestingly, this model has the highest performance \
    on the test set, but strangely its accuracy on the training set is the worst. \
    It has the best calibration histogram, which gives some reliability in its predictions.')])),   
    
    dbc.Row(dbc.Col([html.H3('Deploy')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('In this section, you can prove two of the three models \
    by editing the table and pressing Enter.')])),

    dbc.Row(dbc.Col([input_table],width=12)),
    dbc.Row(dbc.Col([ html.Progress(id="progress_bar", value="0")],width=2)),
    dbc.Row(dbc.Col(id='prediction',children=[]))  

    # Ensemble Model --------------------------------------------------------> #
    
    # dbc.Row(dbc.Col([html.H3('Ensemble')],width=12,
    #                         className="subtitle",style={"textAlign": "left"})),

    # dbc.Row(dbc.Col([html.P('This section shows some correlation metrics taking \
    #                           into account the nature of the predictors, \
    #                          that is, if they are categorical or numerical.')])),
    
    # # Evaluation

    # dbc.Row([dbc.Col([dcc.Graph(figure=cm_train_ensemble)],width=6),dbc.Col([dcc.Graph(figure=pr_train_ensemble)],width=6)]),
    # dbc.Row([dbc.Col([dcc.Graph(figure=pr_cv_ensemble)],width=6),dbc.Col([dcc.Graph(figure=cm_cv_ensemble)],width=6)]),
    # dbc.Row([dbc.Col([dcc.Graph(figure=cm_test_ensemble)],width=6),dbc.Col([dcc.Graph(figure=pr_test_ensemble)],width=6)]),

    # # Model Calibration

    # dbc.Row([dbc.Col([dcc.Graph(figure=calibration_curve_ensemble)],width=8),dbc.Col([dcc.Graph(figure=histogram_ensemble)],width=4)]),

  # Layoud close
  ],className="container")

@app.callback(
    Output('prediction','children'),
    # Input('submit', 'n_clicks'),
    Input('table-editing-simple', 'data'),
    background=True,
    prevent_initial_call=False,
     running=[
        (
            Output("progress_bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        )
     ],
     progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
    )

def prepare_data(set_progress,data):
    # button_clicked = ctx.triggered_id
# if click == 0:
        total = 5
        for i in range(total + 1):
            set_progress((str(i), str(total)))
            time.sleep(1)
        try:
            data_predict = data[0].copy()
            
            for k,v in data_predict.items():
                if k in ['Age','Fare','Pclass','SibSp','Parch','Fare']:
                    data_predict[k] = float(v)     
            X_predict = pd.DataFrame([data_predict])        
            X_predict = transform_instance(preprocessor_imputer,preprocessor_encoder,outlier_,num,cat,scaler,X_predict)
            
            X_predict = X_predict.astype('float64')
            #print('1')
            poly_features = PolynomialFeatures(degree=3, include_bias=False)
            #print('2')
            X_predict_poly = poly_features.fit_transform(X_predict)
            #print('3')
            y_pred_predict_prob_poly =  poly_model.predict_proba(X_predict_poly)[:,1]
            #print('4')
            y_pred_predict_prob_tree = tree_model.predict_proba(X_predict)[:,1]
            #print('5')
            #y_pred_predict_prob_network = (tf.nn.sigmoid(neural_model.predict(X_predict))).numpy().flatten()
            #print('6')   
            #y_pred_predict_network = (y_pred_predict_prob_network >threshold_network).astype(int)
            #print('7')
            y_pred_predict_tree = (y_pred_predict_prob_tree >threshold_tree).astype(int)
            #print('8')
            y_pred_predict_poly = (y_pred_predict_prob_poly >threshold_poly).astype(int)
            #print('9')
            y_prob = [y_pred_predict_prob_poly[0], y_pred_predict_prob_tree[0]] #, y_pred_predict_prob_network]
            #print('10')
            y_fig = [y_pred_predict_poly[0],y_pred_predict_tree[0]] # ,y_pred_predict_network]
            #print('11')
            #print(y_prob)
            #print(y_fig)
            x_fig = ['Polynomial','Ensemble Tree'] #,'Neural Network']
            figure = px.bar(x=x_fig,y=y_fig,hover_data={'Probability':y_prob},title='Prediction Graph',labels={'x':'','y': 'Survive'})
            figure.update_traces(texttemplate='%{y}',textposition='outside')
            figure.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
            figure.update_yaxes(range=(0,1.1),visible=False)
            predict_graph = [dcc.Graph(figure=figure)]
            return predict_graph
        except:
             predict_graph = [html.Br(),html.Div('The input data has an error or is taken by model like atypical data'),html.Br()]
             return predict_graph           

    # if button_clicked == 'submit':
    #     try:
    #         data_predict = data[0].copy()
    #         for k,v in data_predict.items():
    #             if k in ['Age','Fare','Pclass','SibSp','Parch','Fare']:
    #                 data_predict[k] = float(v)     
    #         X_predict = pd.DataFrame([data_predict])        
    #         X_predict = transform_instance(preprocessor_imputer,preprocessor_encoder,outlier_,num,cat,scaler,X_predict)
    #         X_predict = X_predict.astype('float64')

    #         poly_features = PolynomialFeatures(degree=3, include_bias=False)
    #         X_predict_poly = poly_features.fit_transform(X_predict)
    #         y_pred_predict_prob_poly =  poly_model.predict_proba(X_predict_poly)[:,1]
    #         y_pred_predict_prob_tree = tree_model.predict_proba(X_predict)[:,1]
    #         y_pred_predict_prob_network = (tf.nn.sigmoid(neural_model.predict(X_predict))).numpy().flatten()   
    #         y_pred_predict_network = (y_pred_predict_prob_network >threshold_network).astype(int)
    #         y_pred_predict_tree = (y_pred_predict_prob_tree >threshold_tree).astype(int)
    #         y_pred_predict_poly = (y_pred_predict_prob_poly >threshold_poly).astype(int)
    #         y_prob = [y_pred_predict_prob_poly[0], y_pred_predict_prob_tree[0], y_pred_predict_prob_network[0]]
    #         y_fig = [y_pred_predict_poly[0],y_pred_predict_tree[0],y_pred_predict_network[0]]
    #         x_fig = ['Polynomial','Ensemble Tree','Neural Network']
    #         figure = px.bar(x=x_fig,y=y_fig,hover_data={'Probability':y_prob},title='Prediction Graph',labels={'x':'','y': 'Survive'})
    #         figure.update_traces(texttemplate='%{y}',textposition='outside')
    #         figure.update_layout(paper_bgcolor="#0f2537",plot_bgcolor='#0f2537',font={'color':'#ffffff'})
    #         figure.update_yaxes(range=(0,1.1),visible=False)
    #         predict_graph = [dcc.Graph(figure=figure)]
    #         return predict_graph
    #     except:
    #         predict_graph = [html.Br(),html.Div('The input data has an error or is taken by model like atypical data'),html.Br()]
    #         return predict_graph
  

if __name__ == '__main__':
      app.run_server()
