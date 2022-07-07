import re, pandas as pd

train = pd.read_excel("./Data/Data_Train.xlsx")

test = pd.read_excel("./Data/Data_Test.xlsx")

train = train[['Title', 'Author', 'Edition', 'Reviews', 'Ratings','Genre',
               'BookCategory', 'Price']]

test = test[['Title', 'Author', 'Edition', 'Reviews', 'Ratings','Genre',
               'BookCategory']]


authors_1 = list(train['Author'])
authors_2 = list(test['Author'])
authors_1.extend(authors_2)
authorslis = [i.split(",") for i in authors_1]
all_authors = [author.strip().upper() for listin in authorslis for author in listin]

genre_1 = list(train['Genre'])
genre_2 = list(test['Genre'])
genre_1.extend(genre_2)
genre_lis = [i.split(",") for i in genre_1]
all_genres = [genre.strip().upper() for listin in genre_lis for genre in listin]

cat_1 = list(train['BookCategory'])
cat_2 = list(test['BookCategory'])
cat_1.extend(cat_2)
cat_lis = [i.split(",") for i in cat_1]
all_categories = [cat.strip().upper() for listin in cat_lis for cat in listin]

def split_authors(data):
  
  authors = list(data)
  
  A1 = []
  A2 = []
  A3 = []
  A4 = []
  A5 = []
  A6 = []
  A7 = []
  for i in authors:
    
    try :
      A1.append(i.split(',')[0].strip().upper())
    except :
      A1.append('NONE')
      
    try :
      A2.append(i.split(',')[1].strip().upper())
    except :
      A2.append('NONE')
        
    try :
      A3.append(i.split(',')[2].strip().upper())
    except :
      A3.append('NONE')
        
    try :
      A4.append(i.split(',')[3].strip().upper())
    except :
      A4.append('NONE')
        
    try :
      A5.append(i.split(',')[4].strip().upper())
    except :
      A5.append('NONE')
      
    try :
      A6.append(i.split(',')[5].strip().upper())
    except :
      A6.append('NONE')
     
    try :
      A7.append(i.split(',')[6].strip().upper())
    except :
      A7.append('NONE')

      
  return A1,A2,A3,A4,A5,A6,A7
  
def split_genres(data):
  
  genres = list(data)
  
  G1 = []
  G2 = []
  
  for i in genres:
    
    try :
      G1.append(i.split(',')[0].strip().upper())
      
    except :
      G1.append('NONE')
      
    try :
      G2.append(i.split(',')[1].strip().upper())
    except :
      G2.append('NONE')


      
  return G1,G2
  
def split_categories(data):
  
  cat = list(data)
  
  C1 = []
  C2 = []

  for i in cat:
    
    try :
      C1.append(i.split(',')[0].strip().upper())
    except :
      C1.append('NONE')
      
    try :
      C2.append(i.split(',')[1].strip().upper())
    except :
      C2.append('NONE')


      
  return C1,C2
  
def split_edition(data):  
  
  edition  = list(data)
  
  ed_type = [i.split(",– ")[0].strip().upper() for i in edition]
  
  edit_date = [i.split(",– ")[1].strip() for i in edition]
  
  m_y = [i.split()[-2:] for i in edit_date]
  
  for i in range(len(m_y)):
    if len(m_y[i]) == 1:
      m_y[i].insert(0,'NA')
      
  # Based on the given dataset below is the list of possible values for Months
  
  months =  ['Apr','Aug','Dec','Feb', 'Jan', 'Jul','Jun','Mar','May','NA','Nov','Oct','Sep']
  
  ed_month = [m_y[i][0].upper() if m_y[i][0] in months else 'NA' for i in range(len(m_y))]
  ed_year = [int(m_y[i][1].strip()) if m_y[i][1].isdigit() else 0 for i in range(len(m_y))]
  
  return ed_type, ed_month, ed_year


all_categories.append('NONE')
all_genres.append('NONE')
all_authors.append('NONE')



def restructure(data):
  
  
  titles = list(data['Title'])
  
  titles = [title.strip().upper() for title in titles]
  
  
  a1,a2,a3,a4,a5,a6,a7 = split_authors(data['Author']) 
  
  ed_type, ed_month, ed_year = split_edition(data['Edition'])
  

  ratings = list(data['Reviews'])
  ratings = [float(re.sub(" out of 5 stars", "", i).strip()) for i in ratings]
  
  
  reviews = list(data['Ratings'])
  
  plu = ' customer reviews'
  
  reviews = [re.sub(" customer reviews", "", i) if plu in i else re.sub(" customer review", "", i) for i in reviews  ]
  reviews = [int(re.sub(",", "", i).strip()) for i in reviews ]
  

  
  g1, g2 = split_genres(data['Genre'])
  
  c1,c2 = split_categories(data['BookCategory'])

  structured_data = pd.DataFrame({'Title': titles,
                                  'Author1': a1,
                                  'Author2': a2,
                                  'Author3': a3,
                                  'Author4': a4,
                                  'Author5': a5,
                                  'Author6': a6,
                                  'Author7': a7,
                                  'Edition_Type': ed_type,
                                  'Edition_Month': ed_month,
                                  'Edition_Year': ed_year,
                                  'Ratings': ratings,
                                  'Reviews': reviews,
                                  'Genre1': g1,
                                  'Genre2': g2,
                                  'Category1': c1,
                                  'Category2': c2
                                  
                               })
  
  return structured_data



X_train = restructure(train)

Y_train = train.iloc[:, -1].values

X_test = restructure(test)

def unique_items(list1, list2):
  a = list1
  b = list2
  a.extend(b)
  return list(set(a))  

from sklearn.preprocessing import LabelEncoder

le_Title = LabelEncoder()
all_titles = unique_items(list(X_train.Title),list(X_test.Title))
le_Title.fit(all_titles)

le_Edition_Type = LabelEncoder()
all_etypes = unique_items(list(X_train.Edition_Type),list(X_test.Edition_Type))
le_Edition_Type.fit(all_etypes)


le_Edition_Month = LabelEncoder()
all_em = unique_items(list(X_train.Edition_Month),list(X_test.Edition_Month))
le_Edition_Month.fit(all_em)

le_Author = LabelEncoder()
all_Authors = list(set(all_authors))
le_Author.fit(all_Authors)

le_Genre = LabelEncoder()
all_Genres = list(set(all_genres))
le_Genre.fit(all_Genres)

le_Category = LabelEncoder()
all_Categories = list(set(all_categories))
le_Category.fit(all_Categories)

LabelEncoder()

X_train['Title'] = le_Title.transform(X_train['Title'])

X_train['Edition_Type'] = le_Edition_Type.transform(X_train['Edition_Type'])



X_train['Edition_Month'] = le_Edition_Month.transform(X_train['Edition_Month'])

X_train['Author1'] = le_Author.transform(X_train['Author1'])
X_train['Author2'] = le_Author.transform(X_train['Author2'])
X_train['Author3'] = le_Author.transform(X_train['Author3'])
X_train['Author4'] = le_Author.transform(X_train['Author4'])
X_train['Author5'] = le_Author.transform(X_train['Author5'])
X_train['Author6'] = le_Author.transform(X_train['Author6'])
X_train['Author7'] = le_Author.transform(X_train['Author7'])


X_train['Genre1'] = le_Genre.transform(X_train['Genre1'])
X_train['Genre2'] = le_Genre.transform(X_train['Genre2'])


X_train['Category1'] = le_Category.transform(X_train['Category1'])
X_train['Category2'] = le_Category.transform(X_train['Category2'])

X_test['Title'] = le_Title.transform(X_test['Title'])

X_test['Edition_Type'] = le_Edition_Type.transform(X_test['Edition_Type'])



X_test['Edition_Month'] = le_Edition_Month.transform(X_test['Edition_Month'])

X_test['Author1'] = le_Author.transform(X_test['Author1'])
X_test['Author2'] = le_Author.transform(X_test['Author2'])
X_test['Author3'] = le_Author.transform(X_test['Author3'])
X_test['Author4'] = le_Author.transform(X_test['Author4'])
X_test['Author5'] = le_Author.transform(X_test['Author5'])
X_test['Author6'] = le_Author.transform(X_test['Author6'])
X_test['Author7'] = le_Author.transform(X_test['Author7'])


X_test['Genre1'] = le_Genre.transform(X_test['Genre1'])
X_test['Genre2'] = le_Genre.transform(X_test['Genre2'])


X_test['Category1'] = le_Category.transform(X_test['Category1'])
X_test['Category2'] = le_Category.transform(X_test['Category2'])


from sklearn.preprocessing import StandardScaler


sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)


Y_train = Y_train.reshape((len(Y_train), 1)) 

Y_train = sc.fit_transform(Y_train)

Y_train = Y_train.ravel()

from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 123)


print(val_x.shape)




from xgboost import XGBRegressor
import numpy as np


exit()

xgb=XGBRegressor( objective='reg:squarederror', max_depth=6, learning_rate=0.1, n_estimators=100, booster = 'gbtree', n_jobs = -1,random_state = 1)
xgb.fit(train_x,train_y)

y_pred = sc.inverse_transform(xgb.predict(val_x))
y_true = sc.inverse_transform(val_y)

error = np.square(np.log10(y_pred +1) - np.log10(y_true +1)).mean() ** 0.5
score = 1 - error

print("RMLSE Score = ", score)







print("\n\nDONE\n\n")