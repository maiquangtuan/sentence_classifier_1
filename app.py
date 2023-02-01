from transformers import pipeline
import torch
import numpy as np
import onnxruntime
from transformers import AutoTokenizer
import numpy as np
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global tokenizer, session
    
    tokenizer = AutoTokenizer.from_pretrained("model_tokenizer")
    model_path = "mount2_model_classify_sentence_1.onnx"
    session = onnxruntime.InferenceSession(model_path)
# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global tokenizer,session

    model_input = model_inputs.get('input', None)
    if  model_input == None:
        return {'message': "No prompt provided"}
    # Parse out your arguments
    inputs = tokenizer(
        model_input,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="np",
    )
    for key in inputs.keys(): 
        inputs[key]= inputs[key].astype(np.int64)
    # Run the model
    onnx_preds = session.run(None, dict(inputs))[0]
    result = dict(zip(model_input, onnx_preds))

    # Return the results as a dictionary
    return result
