## Part 1 ##

 
The data set included 1,000 positive, and 1,000 negative reviews for a balanced total of 2,000 reviews. 
## Part 2 ##
I trained my classifier with 20 percent of the 2,000 reviews with 200 being positive, and 200 being negative.  
My classifier had a accuracy of 82.25%  
The matrix Accuracy: 82.25%

Confusion Matrix:  
157: Negative reviews correctly predicted as negative.

43: Negative reviews incorrectly predicted as positive.

28: Positive reviews incorrectly predicted as negative.

172: Positive reviews correctly predicted as positive.  
My model uses TF-IDF to see which words are present in a review and how often they occur, but word order is something dosen't comprehend.  
## Part 3 ##  
I tested my model using 25 challenge phrases in five different categories.
- Negation: 2/5 correct (40%)
- Sarcasm: 2/5 correct (40%)
- Mixed Sentiment: 5/5 correct (100%)
- Intensity: 3/5 correct (60%)
- Domain Shift: 4/5 correct (80%)
The overall challenge accuracy was 64%.  

Negation, and Sarcasm were tied for the worst with a score of 40% correct.
Negation is hard to detect since the bag of words model can't see the difference between "not good", and "not bad".  
All the model sees is the words and not their specific order, and assumes that the sentance will be negative even if it's not.
For sarcasm its hard for the model to detect negative vs positive since its common for people to use a positive word to describe a bad thing.  
Sentances like, "Fantastic effects, said no one after seeing them".    
The model will mark it as positive because of the word fantastic even tho that postive word is used to describe something negative.  
 ## Part 4 ##  
 To really work at the negation problem I would see if I could use bigrams to improve my models understanding of negation.   
 Bigrams would allow my model to see the order of the words like, "not bad" as seperate feature which would greatly improve the models accuracy with negation.  
 There is a downside tho, because bigrams make more features more ram and training time is required to make the model work.    
 It would also cost me more time since it is another thing I would have to code into my model.  
 ## Closing ##
 To end off, I would personally not deploy this model because for me the accuracy was not high enough.  
 It was pretty good for the easy sentances, but it got a lot worse for the challanging ones.  
 The main problem areas were sarcasm, and negation with 40% correct answers.  
 Those just so happen to be the most common things used in reviews.  
 Making it easy for me to not want to deploy this model since it would get the common more challanging reviews wrong.  
 Which could bring not true results to the user of this model, and I do not want that at all.  
 To close this paragraph if a human was used for managing the bad reviews the sarcasm, and negation scores would greatly improve.

