# Email_Spam_Filter
This project uses deep neural network model to classify spam messages  and compare the performance with other machine learning model such as Xgboost ,SVM and random forest. This project uses the Enron dataset available here : http://www2.aueb.gr/users/ion/data/enron-spam/.
This approach is combines unsupervised learning with Supervised learning. We will generate the features using TF-IDF algorithm and then use this to features to train Models on labeled enron data.

Model trained and evaluated :

Deep Learning model trained using keras and tensorflow
SVM
Random Forest
XGboost
Deep learning model performs very well on this dataset

Dependencies :

Language - Python 3.5

keras : https://keras.io/

tensorflow : https://www.tensorflow.org/

sklearn: http://scikit-learn.org/stable

numpy : http://www.numpy.org/

pickle: https://docs.python.org/2/library/pickle.html

seaborn: https://seaborn.pydata.org/

To run the model make sure the above dependencies are met and the dataset from [http://www2.aueb.gr/users/ion/data/enron-spam/] is at /home destination. 
For linux environment use the chmod to grant execution permission
 
 sudo chmod 777 spam-filter_classifier.py 

Then type 

 python3 spam-filter_classifier.py     // for Python 3 developemnt

The finding and analysis are available at spam-filter_classifier.pdf.



