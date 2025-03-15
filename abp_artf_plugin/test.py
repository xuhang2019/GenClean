from abp_artf_plugin.inference import Inference
from abp_artf_plugin.pipeline import ProcessingPipeline
import numpy as np


print("### Inference - Testing")
infer = Inference()
segments = np.random.rand(100, 1200)
outputs, item_label, mse_list = infer.fit(segments)
assert np.array(outputs).shape == (100, 1200)

print("### Inference - OK")

print("### Pipeline - Testing")
segments_64Hz= np.random.rand(100, 640)
segments_200Hz = np.random.rand(100, 2000)
sections = np.random.rand(10000)
p = ProcessingPipeline()
outputs, item_label, mse_list = p.process(segments, 120, model_name='PPG')
assert np.array(outputs).shape == (100, 1200), "Expected shape (100, 1200), got " + str(np.array(outputs).shape)

outputs, item_label, mse_list = p.process(segments, 120, model_name='PPG', ml_model='xgboost_feature')
assert np.array(outputs).shape == (100, 14), "XGBoost Feature Test: Expected shape (100, 14), got " + str(np.array(outputs).shape)

outputs, item_label, mse_list = p.process(segments, 120, model_name='PPG', ml_model='ocsvm_feature')
assert np.array(outputs).shape == (100, 14), "OCSVM Feature Test: Expected shape (100, 14), got " + str(np.array(outputs).shape)

outputs, item_label, mse_list = p.process(segments, 120, ckpt_path='abp_artf_plugin/model/input1200_dim20_ppg.ckpt')
assert np.array(outputs).shape == (100, 1200), " Self-defined ckpt_path test, Expected shape (100, 1200), got " + str(np.array(outputs).shape)

outputs, item_label, mse_list = p.process(segments, 120, model_name='PPG', need_process=False)
assert np.array(outputs).shape == (100, 1200), "Need_preprocess test, Expected shape (100, 1200), got " + str(np.array(outputs).shape)


outputs, item_label, mse_list = p.process(segments_64Hz, 64)
assert np.array(outputs).shape == (100, 640), "Expected shape (100, 640), got " + str(np.array(outputs).shape)

outputs, item_label, mse_list = p.process(segments_200Hz, 200)
assert np.array(outputs).shape == (100, 2000), "Expected shape (100, 2000), got " + str(np.array(outputs).shape)

outputs, item_label, mse_list = p.process(sections, 200)
assert np.array(outputs).shape == (10000,), "Expected shape (10000,), got " + str(np.array(outputs).shape)
print("### Pipeline - OK")