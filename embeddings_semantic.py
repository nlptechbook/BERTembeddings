# pip install transformers

import torch
from transformers import BertTokenizer, BertModel

# load pre-trained model tokenizer (vocabulary) and the model:

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Making the model return all hidden-states.
                                  )
# put the model in "evaluation" mode.

model.eval()

# define the sentences to be processed in a list:

sents = []
sents.append('I want an apple.')
sents.append('I want an orange.')
sents.append('I want an adventure.')

# tokenize those sentences 

tokenized_text = []
segments_ids = []
tokens_tensor = []
segments_tensors = []

for i, sent in enumerate(sents):
  tokenized_text.append(tokenizer.encode(sent, add_special_tokens=True))
  segments_ids.append([1] * len(tokenized_text[i]))
  tokens_tensor.append(torch.tensor([tokenized_text[i]]))
  segments_tensors.append(torch.tensor([segments_ids[i]]))

# run our sentences through BERT, and collect all of the hidden states produced in its layers:
 
outputs = []
hidden_states = []
with torch.no_grad():
  for i in range(len(tokens_tensor)):
    outputs.append(model(tokens_tensor[i], segments_tensors[i]))
    # we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states.append(outputs[i][2])


# obtain the embeddings generated for Want in the 12th Encoder layer, and convert them to numpy: 

l12_1 = hidden_states[0][12][0][2][:10].numpy()
l12_2 = hidden_states[1][12][0][2][:10].numpy()
l12_3 = hidden_states[2][12][0][2][:10].numpy()

# compare the obtained vectors to each other to understand how they are close to each other: 

from scipy import spatial

1 - spatial.distance.cosine(l12_1, l12_2)
1 - spatial.distance.cosine(l12_1, l12_3)
1 - spatial.distance.cosine(l12_2, l12_3)

# obtain the initial embeddings for word Want in each sentence:

l0_1 = hidden_states[0][0][0][2][:10].numpy()
l0_2 = hidden_states[1][0][0][2][:10].numpy()
l0_3 = hidden_states[2][0][0][2][:10].numpy()

# divide the Want's embeddings generated in the 12th Encoder layer by the corresponding initial embeddinds:

import numpy as np

l0_12_1 = np.log(l12_1/l0_1)
l0_12_2 = np.log(l12_2/l0_2)
l0_12_3 = np.log(l12_3/l0_3)

# replace NaNs to 0s: 

l0_12_1 = np.where(np.isnan(l0_12_1), 0, l0_12_1)
l0_12_2 = np.where(np.isnan(l0_12_2), 0, l0_12_2)
l0_12_3 = np.where(np.isnan(l0_12_3), 0, l0_12_3)

# calculate the distance between the resulting vectors :

1 - spatial.distance.cosine(l0_12_1, l0_12_2)
1 - spatial.distance.cosine(l0_12_1, l0_12_3)
1 - spatial.distance.cosine(l0_12_2, l0_12_3)

# Determining the most important word using attention weights  
# Here is the 12th layer's attention weights for the first sentence: 'I want an apple.'

outputs[0].attentions[0][0][11].numpy().round(2)

# summing by column, excluding special symbols:

np.sum(outputs[0].attentions[0][0][11].numpy(), axis=0)[1:-1]
