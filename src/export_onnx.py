import torch
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer

def export_model_to_onnx(model_name, output_path, input_names, output_names):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dummy_input = tokenizer("これはテストです。", return_tensors="pt").input_ids

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input_ids": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12,
    )

if __name__ == "__main__":
    model_name = "your-finetuned-model-name"  # Replace with your fine-tuned model name
    output_path = "path/to/your/model.onnx"  # Replace with your desired output path
    input_names = ["input_ids"]
    output_names = ["output"]

    export_model_to_onnx(model_name, output_path, input_names, output_names)