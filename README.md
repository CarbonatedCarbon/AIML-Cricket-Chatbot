# AIML-Cricket-Chatbot
traditional chatbot using aiml with additional features:

The system can answer questions about the sport of cricket. It can provide information on a particular cricket player and summarise a particular cricket series. This information is fetched using an external api.

The program outputs the response through text and audio speech this is done by processing the text on a pre-trained models the generated audio is saved as a .wav file and read to the user. The response can be given through text in multiple languages such as French, dutch, Spanish and English. The text to speech is also supported in each language.    


When the user asked a question not predetermined in the aiml sheet, cosine similarity is used to provide the closest answer in the knowledge base. A csv file containing logical statements on the topic of cricket is given. The user can check if a particular statement contradicts the KB and can add statements to the kb given that it does not contradict it. 

The user is also able to get bowling/batting feedback on a particular cricket pitch based on 5 given pitch condition values. These values are provided to a fuzzy system which decides whether the pitch is ideal for the batsman or bowler.

It can also predict the style of batting action performed in a picture using a convolutional neural network this was done using keras and tensorflow and sci kit learn. The image classification was optimised and improved using the keras tuner library. Making use of the hyperband algorithm. The model predicts the batting action out of 4 possibilities, a drive, leg-glance flick, pullshot and a sweep.
