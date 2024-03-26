import h2o
import pandas as pd
import os
os.chdir(r"C:\Training\Kaggle\Competitions\Santander Customer Satisfaction")
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
h2o.init()

train = h2o.import_file("train.csv", 
                     destination_frame ="train")
test = h2o.import_file("test.csv", destination_frame="test")
print(train.col_names)
all_cols = train.col_names
y = 'TARGET'
X = all_cols[1:-1]

train['TARGET'] = train['TARGET'].asfactor()
print(train['TARGET'].levels())

##################### H2OGradientBoostingEstimator ###########
h2o_gbm = H2OGradientBoostingEstimator(seed=2022)
h2o_gbm.train(x=X, y= y, training_frame=train,
                   model_id="h2o_gbm_santan")

y_pred = h2o_gbm.predict(test_data=test)
y_pred_df = y_pred.as_data_frame()

submit = pd.read_csv("sample_submission.csv")
submit['TARGET'] = y_pred_df['p1']

submit.to_csv("sbt_h2o_gbm.csv", index=False)

##################### H2OGeneralizedLinearEstimator ###########
h2o_glm = H2OGeneralizedLinearEstimator(seed=2022)
h2o_glm.train(x=X, y= y, training_frame=train, 
                   model_id="h2o_glm_santan")

y_pred = h2o_glm.predict(test_data=test)
y_pred_df = y_pred.as_data_frame()

submit = pd.read_csv("sample_submission.csv")
submit['TARGET'] = y_pred_df['p1']

submit.to_csv("sbt_h2o_glm.csv", index=False)

h2o.cluster().shutdown()
