import pickle
from creme import metrics
from helper import transform_text

# Load the model from the file
with open('predict-doc2.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

content = '''
Ordinarily that discretion will be exercised so that costs follow the event and are awarded on a party and party basis. A departure from normal practice to award indemnity costs requires some special or unusual feature in the case: Alpine Hardwood (Aust) Pty Ltd v Hardys Pty Ltd (No 2) [2002] FCA 224
'''

content = transform_text(content)

test_metric = metrics.Accuracy()

print(loaded_model.predict_one(content))
