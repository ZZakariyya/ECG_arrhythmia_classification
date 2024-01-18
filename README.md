# ECG_arrhythmia_classification
This was my first project while I was learning signal processing

1. I downloaded the data from physionet database from MIT and prepared for the model.
2. Model is hybrid form of CNN and RNN.
     a) CNN is really good for 1D data and also it helps me to skip feature extraction
     b) RNN is really nice way of working with times series data which is form of the data I took from physionet.
3. I initially made GRAD-CAM to observe how CNN was training on the data, but after implementing hybrid model, it is kind of unnecessary and less usefull now.
4. Test.py is for testing random ECG images we can find on the internet. It is not working quite well for the time being.
