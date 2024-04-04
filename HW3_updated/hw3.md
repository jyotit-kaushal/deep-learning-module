## Deep Learning Homework 3 - Jyotit Kaushal

**Question 0:** This question is worth no points and serves as a disclaimer. The HW we have prepared here can not be easily transposed and reused to real-life. The quest to design an AI that is capable of predicting the market is a difficult one, and is currently seen as the holy grail of finance. The notions presented in this HW are shown to you, for educational purposes only. Shall you decide to design an AI following ideas discussed in this HW and play with the stock market, SUTD (and yours truly) cannot be held responsible for any losses incurred by the use of this AI on the stock market (seriously, don't).

**Question 1:** Knowing full well of the limitations of the synthetic dataset you are about to play with, do you think the AI we will create and whose job will be to predict the next values of the market will generalize well to real-life markets? Which principle seen in class during W1-W3 explains this generalization issue? Discuss.

**Answer 1:** No I don't think the AI we eventually end up creating to model the stock market and in turn predict the next 5 values of a stock basis the last 20 values will actually be able to generalize well with the real-life market. This is because of this key reason:
- Firstly, with the current dataset, we have created this using a certain mathematical function namely "create_dataset" after having a quick look above. Now by definition we know that neural networks are theoretically able to capture any function that exists with the layers and neurons they possess. Therefore, we might eventually get an AI that does perform really well on this dataset however in the real market, there is things like lulls in the economy, unpredictable inflation, negotiations between different organizations, etc. There is no way an AI will be able to capture that simply by looking at a bunch of numbers.
- The principle of generalization that this case study shows is that of "lack of representative data": ideally with any data science task we should have a dataset truly representative of the population or in this case phenomemon we are studying. Becuase of al the reasons I just mentioned, we in fact do not have a representative data collection and to do so for this task it's something that will be extremely hard.

**Question 2:** Study the code for the Dataset below. How many tensors will be returned when the dataset is summoned using and index *t*, by using the operation *dataset[t]* for instance? What will be the sizes of these tensors? If needed, feel free to play with the Dataset object a bit to find your answer. More importantly, what is the information contained in each of these tensors?

**Answer 2:** Code for the given dataset below, returns us with 3 different tensors, one is a 1D vector with 19 elements, the second 1D vector with 5 elements and lastly one with just one element. What each of these vectors means is that the first with 19 values representas the values of the stock market for instance as a sequence from x(t) to x(t+19)- next the 5 elements represent the 5 elements that can be considered as the target variables and those are what will be predicted as the 5 elements. The last one is meant to serve as an indicator to be the "newest" feature after which we will be predicting the 5 values. 

**Question 3:** For this combination of values, how many samples will the dataset contain? You may need to ask your dataset object using a certain operation. 

**Answer 3:** The number of samples in our dataset are 4096 and we can confirm this using the len() method on our dataset which returns the same. 

**Question 4:** In the code below, there is a *True* value being assigned to the *shuffle* parameter of the dataloader. Until now, we have always used the *shuffle = True* configuration. Would it make sense to change it to a False for the task at hand?

**Answer 4:** Yes it would make sense in fact to change it to False because while for other tasks shuffling might be non-consequential or maybe even preferred to introduce some additional randomness to the data for the model to train on better. However, in the task at hand which is prediciting the next 5 stock values from the previous 20 using what is indeed a time series data we would not want to shuffle because there are certain temporal relations between each of the entries and by shuffling the data points when batching we are essentially getting rid of those which might just be highly consequential in training a good model, therefore we should switch it to False instead. But it is not necessary always even in this case and that's why we are going ahead with it being True.

**Question 5:** If you run the code below, you will get to see the value 16. What does this value 16 correspond to?

**Answer 5:** The value of 6 in this case refers to the number of batches that the pt_dataloader has created and considering we have 4096 samples in our data as earlier seen, by setting the batch_size to be 256 it makes sense that we have 4096/256= 16 batches with 256 entries each.

**Question 6:** What is a Seq2Seq model, and how does it relate to Encoder-Decoder models?

**Answer 6:** A Seq2Seq model as the name suggests essentially helps map one sequence (i.e. the feature sequence) to another different sequence which is your target sequence. An example of sequence-to-sequence model could be a translation model for example. It relates to the encoder-decoder models in the sense that the seq2seq model itself has 2 main components as part of it's architecture. 

An encoder first processes the input sequence using RNNs and basically encodes the input sequence into a context vector with fixed dimensions and we can see this as the semantics of the input sequence. 

The decoder then takes the context vector as generated by the encoder above and again using RNNs produces the output/target sequence one by one to gives us the output sequence. 

Therefore, we can pretty much see the Seq2Seq model as an instance of the Encoder-Decoder models where the overall model is divided into 2 components of and Encoder and a Decoder. 

**Question 7:** This encoder seems to receive all inputs present in the first tensor coming from the Dataloader object, which includes n_inputs - 1 elements (here 20-1 = 19 inputs). This LSTM could then produce 19 outputs, but for some reason, they are not shown on this image. What is the reason for this omission? Why is our diagram suggesting that the final memory vector is the only important information that will come out of this encoder model?

**Answer 7:** The main reason to do so is basically for the clarity of the diagram we are presenting and to essentially focus on only showing the key components and outputs of the encoder which in this case would be the final memory vector as pointed out. It's true yes that the encoder produces an output which is "hidden" upon recieving each of the elements in the sesquence but the focus is still on the main memory vector that is going to be passed through to the decoder. Therefore, this functionality of the encoder is abstracted and the diagram is as it is shown.


**Question 8:** There are a few Nones to be replaced in the code below. Please show your code in your report after you have figured out the correct EncoderRNN class.

**Answer 8:**
```
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size -1 , hidden_size)

    def forward(self, inputs):

        output, hidden = self.lstm(inputs)


        return hidden
```

**Question 9:** Consider the cell below. What is contained in *vec1\[0\]* and *vec2\[0\]*?

**Answer 9:** As seen from our encoder class above, we are returning the output 'hidden' which is actually a tuple containing all the hidden states of the LSTM as well as the cell information. vec1 and vec2 in this case contain just that which is the hidden and cell states of the first layer in the LSTM.


**Question 10:** Assuming that the encoder has seen the inputs $ x(t), x(t+1), ... x(t+18) $, what should we use as a memory vector to play the role of the memory starting point for the decoder?

**Answer 10:** For the memory vector, it would make sense that we take the memory vector to be the hidden state of the encoder at the last time step becuase at this point it has processed al the inputs from x(t) to x(t+18) and this last layer essentially encapsulates the represenation of the entire input sequence, making it the suitable state for the decoder to start initializing it's own internal state. 

**Question 11:** We will use a Decoder that is NOT auto-regressive. What does that mean for the input and output values of our Decoder LSTM-based model?

**Answer 11:** By using a decoder that is NOT auto-regressive it essentially means that by giving it the inptus which is 19(20) in this case- it will produce all the output values again 5 in this case all at once. This is in contrast to producing the output values 1 by 1 and essentially learning from the previous prediction as well. Overall, this architectural choice might cost a bit in overall accuracy of the model but makes up for it by reducing the training time/memory usage when training which is a plus.

**Question 12:** Assuming that the encoder has seen the inputs corresponding to the sample with index $ t $, i.e. $ x(t), x(t+1), ... x(t+18) $, which values should we use in place fo *val1*, *val2*, *val3*, *val4*, *val5*? Remember Q11, we are not planning to use an auto-regressive decoder here. Could you then explain why we only used 19 values as inputs in the encoder part then?

**Answer 12:** One of the reasons for this could be that the necoder uses a shifted window approach meaning that the encoder captures the context for predicting the next 5 values through x(t+1)..x(t+19). In this case then, the val1 in the decoder still corresponds to x(t+1) which is what we're after. 

**Question 13:** Assuming that the encoder has seen the inputs corresponding to the sample with index $ t $, i.e. $ x(t), x(t+1), ... x(t+18) $, what are the target values should we are trying to match with our predictions in place fo *y1*, *y2*, *y3*, *y4*, *y5*?

**Answer 13:** The target values y1...y5 that we should be trying to match with our predictions are the next 5 entries in the sequence of t which is x(t+20)..(x+24)

**Question 14:** What is then the purpose and the expected use for the Linear layer in self.linear? Why is there a for loop in the forward method?

**Answer 14:** The main purpose of the linear layer in the decoder is for a transformation of whatever represenation the decoder layer has created for the next 5 outputs. We want them to have a shape and dimension of 5 of course because that is what our task entails. The linear layer helps us transform it into the final output shape we are looking for. As for the for loop in the forward method, it helps us iterate through some additional internal decoder states and adds them element-wise to the final output before moving on to the linear layer. This is done presumably to help the model learn better.

**Question 15:** Having figured out the questions in Q10-14, can you figure what to use in place of the Nones in the code for the DecoderRNN below? Show your final code in your report.

**Answer 15:**
```
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(1, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)


    def forward(self, outputs, mid, encoder_hidden_states):
        hidden_states = encoder_hidden_states
        final_pred = torch.zeros(outputs.shape).to(outputs.device)
        val = mid
        for i in range(outputs.shape[1]):
            pred, hidden_states = self.lstm(val, hidden_states)
            pred = self.linear(pred)
            final_pred[:, i] = pred.squeeze()
            val = outputs[:, i:i+1]
        return final_pred
```

**Question 16:** Consider the cell below. What should the final size of the *decoder_out* tensor be?

**Answer 16:** torch.Size([256, 5])

**Question 17:** Why have we prefered to use a Decoder-Encoder architecture, instead of a single LSTM that would receive 24 inputs, produce 24 outputs, and would only compare the final 5 predicted values to the ground truth in our dataset?

**Answer 17:** There are two main reasons to do so. Firstly, if we just had an LSTM taking in 24 inputs and producing 24 outputs we would potentially be losing out on a lot of information and interdependencies because by the time we reach the 24th output, the contribution to the learning given by the first few inputs would be really reduced by going through all the layers. Secondly, an LSTM like that would counter intuitively be even more complex which would mean long trianing and times which is not something we are after. 

**Question 18:** Having figured out the models in EncoderRNN and DecoderRNN, can you now figure out the missing code in the cell below? Show it in your report.

**Answer 18:**
```
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder_model = EncoderRNN(self.input_size, self.hidden_size)
        self.decoder_model = DecoderRNN(self.hidden_size, self.output_size)

    def forward(self, inputs, outputs, mid):
        encoder_hidden_states = self.encoder_model(inputs)
        pred_final = self.decoder_model(outputs, mid, encoder_hidden_states)
        return pred_final
```

**Question 19:** Given your understanding of the task, which (very simple) loss function should we use in our trainer function? Show your updated code in your report.

**Answer 19:** We use the very simple MSE loss function in our trainer function as can be seen below:
```
def train(dataloader, model, num_epochs, learning_rate):
    # Set the model to training mode
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, outputs, mid in dataloader:
            # Clear previous gradients
            optimizer.zero_grad()
            # Forward pass
            pred = model(inputs, outputs, mid)
            # Calculate loss
            loss = criterion(pred, outputs)
            total_loss += loss.item()
            # Backward pass and optimization
            loss.backward()
            optimizer.step()          
        
        # Print total loss every few epochs
        if epoch % 25 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {total_loss/len(dataloader)}')
```

**Question 20:** It seems the loss values we are seeing when using the model with randomly initialized parameters is very high. While it seems to decrease, it seems lots of iterations will be needed. The next cell suggests to run the training loop, but initialize the weights of the model using values in the *seq2seq_model_start.pth* file, presumably coming from another roughly similar model, trained on a different but similar task. This is done in an attempt to help the model train better and faster. Under which name is this concept known in Deep Learning?

**Answer 20:** Transfer Leanrning clearly as we are introducing a model which already has some learning of the task at hand as the starting point. 

**Question 21:** The code below shows the predictions produced by your Seq2Seq model after training and can be used to confirm that you have trained the right model! Show some screenshots in your report, and discuss the final performance you have obtained for your model. For your information, I typically obtain an MSE of ~0.05 after 1000 iterations of training. Additional performance can probably be obtained via hyperparameters tuning (changing the size of memory vector, etc.).

**Answer 21:** 
```
Ground truth:  tensor([24.6567, 24.7942, 24.8445, 24.9499, 25.1038])
Prediction:  tensor([24.5019, 24.6671, 24.8061, 24.8588, 24.9641], grad_fn=<SliceBackward0>)
Mean Square Error for Sample:  0.013886986
```
![Results Graph](/Users/jyotit-kaushal/github/deep-learning-module/deeplearningres.png)

