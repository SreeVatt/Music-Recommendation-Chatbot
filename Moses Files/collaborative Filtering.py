import pandas as pd
user=pd.read_csv('Collabrative Filtering T.csv')
user_rating=user['Mukkala Mukkabala']
similar=user.corrwith(user_rating)
corr=pd.DataFrame(similar,columns=['Correlation'])
corr.dropna(inplace=True)
sorted_corr=corr.sort_values(by=['Correlation'],ascending=False)
print(sorted_corr)
