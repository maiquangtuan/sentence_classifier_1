import onnxruntime
from transformers import AutoTokenizer
import numpy as np
input_text = ["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"]
tokenizer = AutoTokenizer.from_pretrained("model_tokenizer")
inputs = tokenizer(
    input_text,
    padding=True,
    truncation=True,
    return_attention_mask=True,
    return_token_type_ids=True,
    return_tensors="np",
)
# print(inputs)
model_path = "mount2_model_classify_sentence_1.onnx"
session = onnxruntime.InferenceSession(model_path)
# print(type(inputs))
# print(dict(inputs))
for key in inputs.keys(): 
    inputs[key]= inputs[key].astype(np.int64)

    # print(type(inputs[key][0][0]))

onnx_preds = session.run(None, dict(inputs))[0]

print(onnx_preds)