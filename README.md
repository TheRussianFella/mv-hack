# mv-hack  

Theese are solutions for MVideo hackathon tasks.

## First task  

First task was to predict rating of products from online store based on reviews users gave them.   
We have tried several methods and final solution is an ensemble of few stacked models.  

## Hackathon task  

*sadly, all code has been lost thanks to universe's evil nature*  
  
Hackathon task was to choose ( from the fixed set ) most important characteristics of products based on reviews, ratings and meta information.  
Solution was to run topic modeling algorithm ( bigARTM ) on reviews data, distinguish n topics ( where n equals to the size of set of characteristics ), take word vectors of most important words in each topic and measure their cosinus similarity to the word vector of characteristic's name - this way for each text we get a percentage of how relevant it is towards each characteristic.    
