import os
import json
import time
import torch
import tarfile
import argparse
import numpy as np
from io import BytesIO
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=2
    )
    return (tokenizer, model)
                     

def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.
    Return a DMatrix (an object that can be passed to predict_fn).
    """
    print("Content type: ", request_content_type)
    if request_content_type == "application/x-npy":        
        stream = BytesIO(request_body)
        return stream.getvalue().decode()
    elif request_content_type == "text/csv":
        return request_body.rstrip("\n")
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )
        

def predict_fn(return_input_fn, return_model_fn):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

    Return a two-dimensional NumPy array (predictions and scores)
    """
    start_time = time.time()
    
    
    print(f"******************** return_input_fn : {return_input_fn}")
    
    
    tokenizer, model = return_model_fn
    encoded_input = tokenizer(return_input_fn, return_tensors='pt')
    
    output = model(**encoded_input)
    pred = torch.argmax(output.logits, dim=1)
    
    print(f"******************** pred : {pred}")
    
    print("--- Inference time: %s secs ---" % (time.time() - start_time))
    return pred


def output_fn(predictions, content_type="application/json"):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    if content_type == "text/csv":
        return predictions.tolist()[0]
    elif content_type == "application/json":
        outputs = json.dumps({'pred': predictions.tolist()[0]})        
        
        return outputs
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))
