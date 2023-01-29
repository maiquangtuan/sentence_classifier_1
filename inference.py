import onnxruntime
from transformers import AutoTokenizer
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
model_path = "mount2_model_classify_sentence_1.onnx"
session = onnxruntime.InferenceSession(model_path)

onnx_preds = session.run(None, dict(inputs))[0]

print(onnx_preds)