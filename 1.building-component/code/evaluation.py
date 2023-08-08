import os
import json
import torch
import tarfile
import argparse
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class preprocess():
    
    def __init__(self, args):
        
        self.args = args
        self.proc_prefix = self.args.proc_prefix #'/opt/ml/processing'
        
        self.input_dir = os.path.join(self.proc_prefix, "input")
        self.model_dir = os.path.join(self.proc_prefix, "model")
        self.output_dir = os.path.join(self.proc_prefix, "output")
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_model(self, ):
        
        model_path = os.path.join(self.model_dir, "model.tar.gz")
        model_dir = "./model"
        
        with tarfile.open(model_path) as tar:
            tar.extractall(path=model_dir)
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=2
        )
    
    def _evaluation(self, ):
        
        test_dataset = load_from_disk(self.input_dir)
                
        metric = {"TP": 0, "FP": 0, "FN":0, "TN":0}
        for idx, (label, content) in enumerate(zip(test_dataset["labels"], test_dataset["content"])):            
            encoded_input = self.tokenizer(content, return_tensors='pt')
            output = self.model(**encoded_input)
            pred = torch.argmax(output.logits, dim=1)

            if pred == 0 and label == 0: metric["TN"] += 1
            elif pred == 1 and label == 1: metric["TP"] += 1
            elif pred == 1 and label == 0: metric["FP"] += 1
            elif pred == 0 and label == 1: metric["FN"] += 1

            if idx % 100 == 0: print (idx)
        
        prec = metric["TP"]/(metric["TP"]+metric["FP"])
        recall = metric["TP"]/(metric["TP"]+metric["FN"])
        f1 = 2*(prec*recall)/(prec+recall)

        print (f'prec: {prec}')
        print (f'recall: {recall}')
        print (f'f1: {f1}')
        
        report_dict = {
            "classification_metrics": {
                "precision": {
                    "value": prec,
                    "standard_deviation": None
                },
                "recall": {
                    "value": recall,
                    "standard_deviation": None
                },
                "f1": {
                    "value": f1,
                    "standard_deviation": None
                },
            },
        }
        
        return report_dict
        
    def execution(self, ):
        
        self._get_model()
        report_dict = self._evaluation()
        
        print("Writing out evaluation report with wer: %f", report_dict)
        evaluation_path = f"{self.output_dir}/evaluation.json"
        with open(evaluation_path, "w") as f:
            f.write(json.dumps(report_dict))
        
        print ("data_dir", os.listdir(self.input_dir))
        print ("model_dir", os.listdir(self.model_dir))
        print ("self.output_dir", os.listdir(self.output_dir))
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_prefix", type=str, default="/opt/ml/processing")
    
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    prep = preprocess(args)
    prep.execution()