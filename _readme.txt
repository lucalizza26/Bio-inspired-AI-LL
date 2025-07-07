
HOW TO VISUALLY SEE THE PERFORMANCE OF THE BEST TRAINED MODEL:

At the bottom of see_model.py, a model name out of the saved ones in the saved_models folder can be selected.
When the code is run, the model will play the lander game 5 times.
note: the game parameters defined at the top of the file can be changed, however, the model is trained on the
shown settings.




HOW THE FILES WERE USED:

trainingV4.py contains the training function used to train the models.

Models are trained with varying parameters in the files like sensitivity_analysis_xx.py.

For each episode, the episode number and reward are saved in the respective folders in csv format.

The results are plotted using the respective plotting codes. They are all very similar.
note: in each of these files, some lines that contain a loop are commented out.
They simply add an heading to the saved csv files.




