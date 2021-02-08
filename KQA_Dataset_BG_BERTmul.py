from utils.korquad_utils_revised import read_squad_examples
from torch.utils.data import TensorDataset
import torch
import pickle
import os
import logging
logger = logging.getLogger(__name__)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
import warnings
warnings.filterwarnings(action='ignore')



def convert_data_to_features(data,seq_len):
    features=[]
    for single_data in data:
        single_feature={}
        doc=single_data.doc_tokens
        question=single_data.question_text
        answer=single_data.orig_answer_text
#         if((single_data.start_position==0) or (single_data.end_position==0)):
#             continue
        answer2=doc[single_data.start_position:single_data.end_position+1][0]
        tokenized_features=tokenizer(question," ".join(doc))
        tokenized_doc=tokenizer.tokenize(" ".join(doc))
        tokenized_answer=tokenizer.tokenize(answer)
        tokenized_answer2=tokenizer.tokenize(answer2)
        featurized_text=tokenizer.decode(tokenized_features['input_ids'])
        tokenized_featurized_text=tokenizer.tokenize(featurized_text)

        doc_tokenized=tokenizer.tokenize(" ".join(doc))
                   
        for i in range(len(doc_tokenized)):
            if doc_tokenized[i:i+len(tokenized_answer2)] == tokenized_answer2:
                start_position=i+len(tokenizer.tokenize(single_data.question_text))+2
                end_position=i+len(tokenized_answer2)+len(tokenizer.tokenize(single_data.question_text))+2
                for j in range(len(doc_tokenized[i:i+len(tokenized_answer2)])):
                    if(doc_tokenized[i:i+len(tokenized_answer2)][j:j+len(tokenized_answer)]==tokenized_answer):
                        start_position+=j
                        end_position=start_position+len(tokenized_answer)

            
        
        
        
#         for i in range(len(tokenized_featurized_text)):
#             if tokenized_featurized_text[i:i+len(tokenized_answer2)] == tokenized_answer2:
#                 start_position=i
#                 end_position=i+len(tokenized_answer)
#                 if tokenized_featurized_text[i:i+len(tokenized_answer)] == tokenized_answer:
#                     start_position=i
#                     end_position=i+len(tokenized_answer)
#                     break

                    
                    
        
#         for i in range(len(tokenized_featurized_text)):
#             if tokenized_featurized_text[i:i+len(tokenized_answer)] == tokenized_answer:
#                 start_position=i
#                 end_position=i+len(tokenized_answer)
#                 if tokenized_featurized_text[i:i+len(tokenized_answer2)] == tokenized_answer2:
#                     start_position=i
#                     end_position=i+len(tokenized_answer)
#                     break
        for i in range(seq_len-len(tokenized_features['input_ids'])):
            tokenized_features['input_ids'].append(0)
            tokenized_features['token_type_ids'].append(0)
            tokenized_features['attention_mask'].append(0)
            
        if(len(tokenized_features['input_ids'])!=seq_len):
            continue
        if(len(tokenized_features['token_type_ids'])!=seq_len):
            continue
        if(len(tokenized_features['attention_mask'])!=seq_len):
            continue
        single_feature['tokenized_features']=tokenized_features
        single_feature['start_position']=start_position
        single_feature['end_position']=end_position
        features.append(single_feature)
                    
    return features




def load_and_cache_dataset(input_file, max_seq_length=512,testset_ratio=0.1):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(os.path.dirname(input_file), '_cached_bertmul_{}_{}'.format('train',str(max_seq_length)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
                                        
                                        
                                        
                                        
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        data = read_squad_examples(input_file, is_training=True, version_2_with_negative=False)
        features = convert_data_to_features(data,max_seq_length)
                                        
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f['tokenized_features']['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['tokenized_features']['attention_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['tokenized_features']['token_type_ids'] for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f['start_position'] for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f['end_position'] for f in features], dtype=torch.long)
    
    train_len=int(len(all_input_ids)*(1-testset_ratio))
    
    train_dataset = TensorDataset(all_input_ids[:train_len], all_input_mask[:train_len], all_segment_ids[:train_len], all_start_positions[:train_len], all_end_positions[:train_len])
    test_dataset = TensorDataset(all_input_ids[train_len:], all_input_mask[train_len:], all_segment_ids[train_len:], all_start_positions[train_len:], all_end_positions[train_len:])
    return train_dataset, test_dataset