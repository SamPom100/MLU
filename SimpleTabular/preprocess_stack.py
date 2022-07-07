import autogluon.text, autogluon.tabular, pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset

train = TabularDataset(data="./data/training.csv")
test = TabularDataset(data="./data/mlu-leaderboard-test.csv")

train.drop(['Synopsis'], axis=1, inplace=True)
test.drop(['Synopsis'], axis=1, inplace=True)



authors_1 = list(train['Author'])
authors_2 = list(test['Author'])

authors_1.extend(authors_2)

authorslis = [i.split(",") for i in authors_1]

    
all_authors = [author.strip().upper() for listin in authorslis for author in listin]

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
  
all_authors.append('NONE')

genre_1 = list(train['Genre'])
genre_2 = list(test['Genre'])

genre_1.extend(genre_2)

genre_lis = [i.split(",") for i in genre_1]

      
all_genres = [genre.strip().upper() for listin in genre_lis for genre in listin]

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
  
all_genres.append('NONE')

cat_1 = list(train['BookCategory'])
cat_2 = list(test['BookCategory'])

cat_1.extend(cat_2)

cat_lis = [i.split(",") for i in cat_1]


all_categories = [cat.strip().upper() for listin in cat_lis for cat in listin]

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
  
all_categories.append('NONE')

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


import re


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
  p1 = list(data['Price'])
  i1 = list(data['ID'])
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
                                  'Category2': c2,
                                  'Price': p1,
                               })
  return structured_data

def restructure_TEST(data):
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
  i1 = list(data['ID'])
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
                                  'Category2': c2,
                                  'ID':i1,
                               })
  return structured_data


X_train = restructure(train)
X_test = restructure_TEST(test)

#BUILD MODEL
predictor = TabularPredictor(label="Price").fit(
    X_train,
    auto_stack=True,
)

#predictor = TabularPredictor.load("AutogluonModels/ag-20220617_161654/")

#BUILD PREDICTIONS
prediction = predictor.predict(X_test)

submission = X_test[["ID"]].copy(deep=True)
submission["Price"] = prediction
submission.to_csv(
    "./data/predictions/newMethodStackwithID2.csv",
    index=False,
)