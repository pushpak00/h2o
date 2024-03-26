import h2o
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
h2o.init()

df = h2o.import_file("Bankruptcy.csv", 
                     destination_frame ="brupt")

print(df.col_names)
all_cols = df.col_names
y = 'D'
X = all_cols[3:]

df['D'] = df['D'].asfactor()
print(df['D'].levels())


train,  test = df.split_frame(ratios=[.7],seed=2022,
                              destination_frames=['train','test'])
print(df.shape)
print(train.shape)
print(test.shape)

h2o_rf = H2ORandomForestEstimator(seed=2022)
h2o_rf.train(x=X, y= y, training_frame=train, 
                   validation_frame=test, 
                   model_id="h2o_rf_brupt")

print(h2o_rf.auc(valid=True) )
print(h2o_rf.confusion_matrix() )

y_pred = h2o_rf.predict(test_data=test)
y_pred_df = y_pred.as_data_frame()

print(h2o_rf.logloss(train=False, valid=True, xval=False))

print(h2o_rf.model_performance())

print(h2o_rf)

##################### H2OGradientBoostingEstimator ###########
h2o_gbm = H2OGradientBoostingEstimator(seed=2022)
h2o_gbm.train(x=X, y= y, training_frame=train, 
                   validation_frame=test, 
                   model_id="h2o_gbm_brupt")

print(h2o_gbm.auc(valid=True) )
print(h2o_gbm.confusion_matrix() )

y_pred = h2o_gbm.predict(test_data=test)
y_pred_df = y_pred.as_data_frame()

print(h2o_gbm.logloss(valid=True))

##################### H2OGeneralizedLinearEstimator ###########
h2o_glm = H2OGeneralizedLinearEstimator(seed=2022)
h2o_glm.train(x=X, y= y, training_frame=train, 
                   validation_frame=test, 
                   model_id="h2o_glm_brupt")

print(h2o_glm.auc(valid=True) )
print(h2o_glm.confusion_matrix() )

y_pred = h2o_glm.predict(test_data=test)
y_pred_df = y_pred.as_data_frame()

print(h2o_glm.logloss(valid=True))


h2o.cluster().shutdown()
