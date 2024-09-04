# AIML-Cricket-Chatbot
traditional chatbot using aiml with additional features:



The system can answer questions about the sport of cricket. 
![image](https://github.com/user-attachments/assets/8df85697-9a99-4ac1-a2ad-ce082d701999)


It can provide information on a particular cricket player and summarise a particular cricket series. This information is fetched using an external api.

![image](https://github.com/user-attachments/assets/9d35aace-7467-4807-a74b-49f0fcd98bd3)

![image](https://github.com/user-attachments/assets/efc32f41-b1c9-414d-82f2-39d092a6ae02)


The program outputs the response through text and audio speech this is done by processing the text on a pre-trained models the generated audio is saved as a .wav file and read to the user. The response can be given through text in multiple languages such as French, dutch, Spanish and English. The text to speech is also supported in each language.    
![image](https://github.com/user-attachments/assets/09a37cf2-d102-4adb-842f-6a7b98d04cc9)


When the user asked a question not predetermined in the aiml sheet, cosine similarity is used to provide the closest answer in the knowledge base. A csv file containing logical statements on the topic of cricket is given. The user can check if a particular statement contradicts the KB and can add statements to the kb given that it does not contradict it. 

![image](https://github.com/user-attachments/assets/db33dd8f-68d3-45b7-a680-fcb2181cd56c)


The user is also able to get bowling/batting feedback on a particular cricket pitch based on 5 given pitch condition values. These values are provided to a fuzzy system which decides whether the pitch is ideal for the batsman or bowler.

![image](https://github.com/user-attachments/assets/a958721e-08aa-4894-8fb3-c851615d148b)


It can also predict the style of batting action performed in a picture using a convolutional neural network this was done using keras and tensorflow and sci kit learn. The image classification was optimised and improved using the keras tuner library. Making use of the hyperband algorithm. The model predicts the batting action out of 4 possibilities, a drive, leg-glance flick, pullshot and a sweep.
![image](https://github.com/user-attachments/assets/19492c37-7e89-4e33-aefb-3bae16af0da4)
![image](https://github.com/user-attachments/assets/73dec70e-8d7a-4d0d-ba00-43030994b292)
![image](https://github.com/user-attachments/assets/4db52a80-27f0-4e71-b190-34d7b58bd767)


SYSTEM REQUIREMENTS:
the project was run using: 
-WSL 2
-python 3.10.12
