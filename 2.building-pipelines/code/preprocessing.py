import os
import pickle
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

class preprocess():
    
    def __init__(self, args):
        
        self.args = args
        self.proc_prefix = self.args.proc_prefix #'/opt/ml/processing'
        
        self.input_dir = os.path.join(self.proc_prefix, "input")
        self.output_dir = os.path.join(self.proc_prefix, "output")
        self.split_rate = float(self.args.split_rate)
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "test"), exist_ok=True)
        
        tokenizer_name = 'distilbert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
#     def to_pickle(self, obj, path):
        
#         with open(path, "wb") as fw: pickle.dump(obj, fw)
    
#     def load_pickle(self, path):
        
#         with open(path, "rb") as fr:
#             obj = pickle.load(fr)
        
#         return obj
    
    def tokenize(self, batch):
        return self.tokenizer(batch['content'], padding="longest", truncation=True) #'max_length'
        
    def preprocess(self, train, test):

        # Helper function to get the content to tokenize        
        train = Dataset.from_pandas(train)
        test = Dataset.from_pandas(test)
        
        # Tokenize
        train = train.map(self.tokenize, batched=True, batch_size=len(train))
        test = test.map(self.tokenize, batched=True, batch_size=len(test))

        # Set the format to PyTorch
        train = train.rename_column("label", "labels")
        train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        test = test.rename_column("label", "labels")
        test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        print (train.data)
        
        return train, test
    
    def execution(self, ):
        
        data = pd.read_csv(os.path.join(self.input_dir, "amazon_polarity.csv"))
        data = data.sample(frac=1).reset_index(drop=True) ## shuffle
        train, test = data.iloc[:int(data.shape[0]*self.split_rate), :], data.iloc[int(data.shape[0]*self.split_rate):, :]
        print (f'train: {train.shape}, test: {test.shape}')
        
        train, test = self.preprocess(train, test)
        
        train.save_to_disk(
            dataset_path=os.path.join(self.output_dir, "train")
        )
        test.save_to_disk(
            dataset_path=os.path.join(self.output_dir, "test")
        )
        
        # self.to_pickle(
        #     obj=train,
        #     path=os.path.join(self.output_dir, "train", "train.pkl")
        # )
        # self.to_pickle(
        #     obj=test,
        #     path=os.path.join(self.output_dir, "test", "test.pkl")
        # )
        
        print ("data_dir", os.listdir(self.input_dir))
        print ("self.output_dir", os.listdir(self.output_dir))
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_prefix", type=str, default="/opt/ml/processing")
    parser.add_argument("--split_rate", type=str, default="0.5")
    
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    prep = preprocess(args)
    prep.execution()