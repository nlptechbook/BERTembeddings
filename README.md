# Discovering trends in BERT embeddings of different levels for the task of semantic context determining

Talking about contextual embeddings in BERT, we mean the hidden states of a pre-trained model. To start with, however, BERT uses non- contextual, pre-trained (static) embeddings being taken from the look-up table. It happens in the Embedding layer that comes before the Encoder layers, which, then, generate the hidden states for the sequence being processed. That is, contextual embedding comes from non-contextual and in some sense it is a product of diffusion of embeddings of neighboring tokens. 

## Hypothesis
   
The hypothesis is that the difference between the initial, non-contextual embedding of a token and the contextual embedding generated then with the encoders for this same token can contain information about the context itself and, therefore, can be useful when, for example, you need to generate a summary embedding for the entire sentence in such tasks as, say, sentence entailment.

## Intuition 
    
A token's embedding is nothing but a numerical vector that represents that token in a n-dimensional space. Each dimension holds some semantic information of that token. The concept can be easily understood with the following simplified example. Suppose the features are Banking, Postal, Gaming. Then, the word Card could be represented with the following 3-dimensional vector: (0.85, 0.6, 0.45). As you might guess, this is a non- contextual representation of the word Card, which shows just the probabilities of appearing that word in a certain context. Now suppose the word Card appears in the following phrase: 

> I sent her a postal card. 

So the contextual representation of Card might look now as follows: (0.2, 1, 0.1), increasing the value related to Postal and decreasing the values related to the other categories. To see the change between the original vector and the new one by feature, you might divide (element wise) the latter vector by the former, using the logarithm to secure a smooth approximation: 

> c = log(b/a)

> where a - non-cotextual (initial) vector,  
>       b - cotextual vector

In this particular example, the resulting vector of the above calculation will look as follows: 

> (-1.4, 0.5, -1.5)

This new representation gives you a clear idea of the change in each feature, either positive or negative. You may be wondering, why do I need this representation at all - why not use just the contextual vector instead? The idea is that knowing the changes in feature values (rather than their absolute values) can be more helpful to the model when the absolute values do not change much. This happens when it's not immediately clear which feature dominates and which should be paid attention first in the current context. This can be best understood by example. Continuing with our card example, consider the following two sentence passage: 

 > We're going to send her a card.   
 > Where is the nearest mailbox? 

The first sentence doesn't tell you explicitly what kind of card is meant. However, the transitive verb Send used here allows your model to make a cautious guess that a postal card is meant. So, based on the context of the first sentence, the transition of the card's initial embedding to a contextual one might look as follows, increasing a bit the values of the Postal feature and decreasing all the others:

> (0.85, 0.6, 0.45) -> (0.8, 0.75, 0.4)

As you can see, however, the value of the Banking feature (comes first) is still greater than the other values, which would make your model treat it as a priority feature when deciding on the context. In contrast, considering the changes in feature values would be more informative, since it explicitly shows what weights increase and what decrease:   

> log((0.8, 0.75, 0.4)/(0.85, 0.6, 0.45)) -> (-0.1,  0.2, -0.1)

In this particular example, the context of the second sentence is explicitly related with postage. So, in the task of sentence entailment, the proposed approach would help your model make a correct prediction.  

## Experimenting with BERT embeddings

Unlike the above toy example, real models typically use several hundred dimensions for embedding. For example, the base BERT models use 768 dimensional space for embedding, where each dimension is not associated with an explicitly named semantic category. However, the main idea remains the same: if two embeddings have high values in the same dimension, it indicates that their corresponding words have a connection with a certain, one and the same semantic category, such as Banking, Gaming, etc.  

Let's experiment with BERT embeddings, using simple and clear example. Consider the following three sentences: 

> I want an apple.    
> I want an orange.   
> I want an adventure.

In each of the above sentences, we use the same transitive verb Want. But in the first two sentences, we use the direct objects: Apple and Orange which belong to the same semantic category: Fruit. The third sentence uses the direct object: Adventure that obviously belongs to another category.

### The purpose of our experiment: 

1. Checking how the difference in the semantics of the direct object affects the embedding of the transitive verb. 
2. If there is such an effect, making it more clearly expressed. 

### Implementation

Google Colab: https://colab.research.google.com/drive/1k_R1qOS79auwS2JEJ7D1mYMXHXad29fd?usp=sharing

To get pre-trained BERT models, we'll use transformers from Hugging Face:   

> pip install transformers

We'll need the BERT tokenizer and the bare Bert Model transformer:  
```python
import torch
from transformers import BertTokenizer, BertModel
```
Then, we can load pre-trained model tokenizer (vocabulary) and the model:
```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Making the model return all hidden-states.
                                  )
```                                  
We put the model in "evaluation" mode.
```python
model.eval()
```
Below we define the sample sentences to be used in our example:
```python
sents = []
sents.append('I want an apple.')
sents.append('I want an orange.')
sents.append('I want an adventure.')
```
Next, we need to tokenize those sentences: 
```python
tokenized_text = []
segments_ids = []
tokens_tensor = []
segments_tensors = []

for i, sent in enumerate(sents):
  tokenized_text.append(tokenizer.encode(sent, add_special_tokens=True))
  segments_ids.append([1] * len(tokenized_text[i]))
  tokens_tensor.append(torch.tensor([tokenized_text[i]]))
  segments_tensors.append(torch.tensor([segments_ids[i]]))
```
After that, we can run our sentences through BERT, and collect all of the hidden states produced in its layers:
```python 
outputs = []
hidden_states = []
with torch.no_grad():
  for i in range(len(tokens_tensor)):
    outputs.append(model(tokens_tensor[i], segments_tensors[i]))
    # we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states.append(outputs[i][2])
```
So, we have the output of the Embedding layer + the output of all the 12 Encoder layers for each sentence.
```python
len(hidden_states[0])
13
```
Each layer has 768 dimensions.
```python
len(hidden_states[0][12][0][2])
768
```
For simplicity here, we'll take into account only the first 10 values of the embedding:
```python
hidden_states[0][12][0][2][:10].numpy()
```
>array([ 0.44462553,  0.21318859,  1.1400639 , -0.05000957,  0.43685108,
>        0.91370475, -0.6992555 ,  0.13507934, -0.42180806, -0.66882026],
>      dtype=float32)

In each sentence in this example, we're interested only in the second word (Want). Below we obtain the embeddings generated for Want in the 12th Encoder layer, and convert them to numpy: 
```python
l12_1 = hidden_states[0][12][0][2][:10].numpy()
l12_2 = hidden_states[1][12][0][2][:10].numpy()
l12_3 = hidden_states[2][12][0][2][:10].numpy()
```
It would be interesting to compare the obtained vectors to each other to figure out how they are semantically close to each other: 
```python
from scipy import spatial

1 - spatial.distance.cosine(l12_1, l12_2)
0.9869935512542725

1 - spatial.distance.cosine(l12_1, l12_3)
0.8980972170829773

1 - spatial.distance.cosine(l12_2, l12_3)
0.8874450922012329
```
As you can see (and might expect), the embeddings for Want in the first two sentences are much closer to each other than any of them to the embedding of Want that occurred in the third sentence.  

Lets now check if we can obtain a clearer picture of the differences in the embeddings, reflecting the semantic difference arising from the context.  

To start with, we need to obtain the initial embeddings for word Want in each sentence:
```python
l0_1 = hidden_states[0][0][0][2][:10].numpy()
l0_2 = hidden_states[1][0][0][2][:10].numpy()
l0_3 = hidden_states[2][0][0][2][:10].numpy()
```
we can now divide the Want's embeddings generated in the 12th Encoder layer by the corresponding initial embeddinds:
```python
import numpy as np

l0_12_1 = np.log(l12_1/l0_1)
l0_12_2 = np.log(l12_2/l0_2)
l0_12_3 = np.log(l12_3/l0_3)
```
Before proceeding we need to replace NaNs to 0s: 
```python
l0_12_1 = np.where(np.isnan(l0_12_1), 0, l0_12_1)
l0_12_2 = np.where(np.isnan(l0_12_2), 0, l0_12_2)
l0_12_3 = np.where(np.isnan(l0_12_3), 0, l0_12_3)
```
Let's now calculate the distance between the resulting vectors to understand whether these new representations can better indicate the semantic difference between the underlying words:
```python
1 - spatial.distance.cosine(l0_12_1, l0_12_2)
0.9640171527862549

1 - spatial.distance.cosine(l0_12_1, l0_12_3)
0.4167512357234955

1 - spatial.distance.cosine(l0_12_2, l0_12_3)
0.3458264470100403
```
Comparing with the similarity results obtained earlier for the embeddinds generated with the 12th layer, we may conclude that these new representations enable us to get a clearer understanding of how the underlying words are different from each other depending on the context they are in. 

### Attention Weights to Choose the Most Important Word in Terms of Context

As you no doubt have realized, the general idea is that the vector resulting from dividing the contextual embedding of a token by the static embedding of this same token includes information about the context of the entire sentence. The previous example illustrated how the proposed method works when applied to a transitive verb of a sentence. The question arises: Does it always have to be a transitive verb and is one word per sentence sufficient for this kind of analysis? 

Well, it is intuitively clear that it should be an important word in terms of context. To choose one in a particular sentence, you can take advantage of attention weights generated in the encoder layers. Here is the code:

Before going any further, let's look at the tokens for which we're going to obtain the matrix of attention weights:
```python
tokenizer.convert_ids_to_tokens(tokenized_text[0])
```
> ['[CLS]', 'i', 'want', 'an', 'apple', '.', '[SEP]']

Here is the 12th layer's attention weights for the sentence: 'I want a green apple.'
```python
outputs[0].attentions[0][0][11].numpy().round(2)

array([[0.93, 0.02, 0.  , 0.01, 0.  , 0.  , 0.03],
       [0.3 , 0.05, 0.24, 0.07, 0.14, 0.06, 0.15],
       [0.38, 0.41, 0.04, 0.02, 0.06, 0.02, 0.07],
       [0.48, 0.11, 0.16, 0.02, 0.02, 0.04, 0.17],
       [0.07, 0.07, 0.26, 0.27, 0.06, 0.05, 0.23],
       [0.52, 0.05, 0.06, 0.04, 0.07, 0.  , 0.26],
       [0.71, 0.06, 0.03, 0.03, 0.01, 0.  , 0.15]], dtype=float32)
```
We sum by column, excluding special symbols:

```python
np.sum(outputs[0].attentions[0][0][11].numpy(), axis=0)[1:-1]

array([0.7708196 , 0.7982767 , 0.45694995, 0.36948416, 0.17060593],
      dtype=float32)  
``` 
According to the above, the second word (Want) is the most important one in the sentence.

### Embeddings of how many tokens in a sentence need to be used

As stated, we're getting a single vector - derived for the most important word in a sentence - that includes information about the context of the entire sentence. However, to get a better picture of the sentence context, it would also be nice to have such a vector for the word that is most syntactically related to that most important word. Why do we need this? 
 
A simple analogy from life can help answer this question: If you admire the surrounding beauties while sitting, say, in a restaurant located inside the tower - the views you contemplate will not include the tower itself. To take a photo of the tower view, you first need to exit the tower. 

Now, how can we determine the word that is most syntactically related to the most important word in the sentence? (best place to take a photo of the tower, according to the previous analogy) The answer is: with the help of the attention weights described in the previuos section. Below we are determining the word that is syntactically closest to the most important word (Want, in this particular example). For that we check the attention weights in all 12 layers. To start with, we create an empty array:
```python
a = np.empty([0, len(np.sum(outputs[0].attentions[0][0][11].numpy(), axis=0)[1:-1])])
```
Next, we fill in the matrix of attention weights:
```python
for i in range(12):
  a = np.vstack([a,np.sum(outputs[0].attentions[0][0][i].numpy(), axis=0)[1:-1]])
```
We are not interested in the punctuation symbol. So, we'll remove the column in the vector:
```python
a = np.delete(a, -1, axis=1)
```
Let's now determine in which layers Want drew the most attention:
```python
print(np.argmax(a,axis=1))
b = a[np.argmax(a,axis=1) == 1]

array([1, 3, 0, 2, 1, 1, 3, 0, 3, 3, 1, 1])
```
Next, we can determine which token draws more attention after Want in the layers where Want is in the lead:
```python
c = np.delete(b, 1, axis=1)
d = np.argmax(c, axis =1)
print(d)
counts = np.bincount(d)
print(np.argmax(counts))

[0 2 2 2 0]
2
```
So, in this particular example, we have word Apple as the one that is the most syntactically related to word Want. This is quite expected because these words represent the direct object and the transitive verb, respectively. 
```python
_l12_1 = hidden_states[0][12][0][4][:10].numpy()
_l0_1 = hidden_states[0][0][0][4][:10].numpy()
_l0_12_1 = np.log(_l12_1/_l0_1)
_l0_12_1 = np.where(np.isnan(_l0_12_1), 0, _l0_12_1)
```
Let's now compare the vectors derived from the embeddings of words Apple and Want.
```python
print(_l0_12_1)

array([ 3.753544  ,  1.4458075 , -0.56288993, -0.44559467,  0.9137548 ,
        0.33285233,  0.        ,  0.        ,  0.        ,  0.        ],
      dtype=float32)
      
print(l0_12_1)

array([ 0.        ,  0.        ,  0.        ,  0.        , -0.79848075,
        0.6715901 ,  0.30298436, -1.6455574 ,  0.1162319 ,  0.        ],
      dtype=float32)
```
As you can see, one of the values in the pair of matching elements in the above two vectors is in most cases zero while the other value is non-zero - i.e. the vectors look complementary (Remember the tower view analogy: Neighboring sights are visible from the tower, but in order to see the tower itself - perhaps the main attraction - you need to leave it) So, you can safely sum up these vectors elementwise to combine the available information into a single vector. 
```python
s = _l0_12_1 + l0_12_1
print(s)

array([ 3.753544  ,  1.4458075 , -0.56288993, -0.44559467,  0.11527407,
        1.0044425 ,  0.30298436, -1.6455574 ,  0.1162319 ,  0.        ],
      dtype=float32)
```
The above vector can next be used as input for sentence-level classification. 
