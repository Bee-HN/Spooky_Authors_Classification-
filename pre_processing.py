import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('train.csv')

#author_names = {'EAP':'Edgar Allen Poe', 'MWS':'Mary Shelly', 'HPL':'HP Lovecraft'}

#plotly.tools.set_credentials_file(username='hng100289', api_key='dHePSQcOYvgKcPtpsJIU')
#data = [go.Bar( x = train.author.map(author_names).unique(),
#                y = train.author.value_counts().values,
#                marker = dict (colorscale = 'Jet', color= train.author.value_counts().values),
#                text='Text entries attribute to authors')]
#layout = go.Layout(title = 'Data Distribution', xaxis = dict(title='Authors'), yaxis = dict(title='Count'))

#fig = go.Figure(data=data, layout=layout)
#py.iplot(fig, filename='Data Distribution Graph')

#all_words = train['text'].str.split(expand=True).unstack().value_counts()

#data = [go.Bar(x = all_words.index.values[:100],
#               y = all_words.values[:100],
#               marker= dict(colorscale='Jet', color=all_words.values[:100]),
#               text='Words Counts')]

#layout = go.Layout(title='Most Common Words In The Dataset (Pre-Processing)', xaxis = dict(title='Most Common Words'), yaxis = dict(title='Count'))

#fig = go.Figure(data=data, layout=layout)
#py.plot(fig, filename='Most Common Words')

#Create three lists of documents - one for each author

eap = train[train.author == 'EAP']['text'].values
hpl = train[train.author == 'HPL']['text'].values
mws = train[train.author == 'MWS']['text'].values


#Get the text data from train dataframe
text = list(train.text.values[:500])

vectorizer = TfidfVectorizer(stop_words='english', min_df=0, decode_error='ignore')

text_transform = vectorizer.fit_transform(text)

clf = svm.SVC(probability=True)
target = train.author.values[:500]


X_train, X_test, Y_train, Y_Test = cross_validation.train_test_split(text_transform, target, test_size=0.1)

clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_Test)

KNNC = KNeighborsClassifier(n_neighbors=10, metric='cosine')

KNNC.fit(X_train,Y_train)

accuracy = KNNC.score(X_test, Y_Test)
print(accuracy)






