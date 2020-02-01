


import pandas as pd
from tqdm import tqdm
import numpy as np
from bisect import bisect
import glob
import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures as plnm
from sklearn.linear_model import LinearRegression as lrg
from sklearn.linear_model import LogisticRegression as lgr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')




df_raw = pd.read_excel('Data.xlsx')

print(df_raw.head())




def fakebidder(data,itera,issue):#itera means simulated times for one bidder
    
    #create fake bids
    P_C = []
    prob = []
    v = []
    cnt = []
    betw = []
    data=data[data.issueid==issue]
    data=data[data.price>0]
    
    
    
    df = pd.DataFrame()

    P_C = []
    prob = []
    v = []
    cnt = []
    betw = []
    
    P_market =  data.price_cutoff.iloc[0,]
    Q = data.ntfdamt.iloc[0,]
   
    bidders = data.loc[:,['member_code','pddum']].drop_duplicates()
    no_bids = data.shape[0]
    no_bidders = len(bidders)
    
    #differentiate big and small bidder
    b_bidders = bidders[bidders.pddum == 1].member_code.tolist()
    s_bidders = bidders[bidders.pddum == 0].member_code.tolist()
    
    count = 0
    
    for bidder in tqdm(bidders.member_code.tolist()):
        
        #select bidders randomly
        if bidder in b_bidders:   
            b_to_select =  list(set(b_bidders) - set([bidder]))
            s_to_select = s_bidders
            
        else:
            b_to_select =  b_bidders
            s_to_select = list(set(s_bidders) - set([bidder]))


        # fixed the current bidder
        fixed0 = data[data.member_code == bidder]
        fixed0 = fixed0.sort_values('price',ascending = False)
        qc = fixed0.allocatedamount.sum()
        #fixed0=fixed0.sort_values('allocatedamount',ascending = True)
        
        
        fixed0['Q_C'] = qc

        
        
        
        pclist=[]
        pc_sum= 0
        qc_i=0
    
    
        while(count < itera):
            re_df=pd.DataFrame()
            #create fake auctions
            random_bidders = np.concatenate((np.random.choice(b_to_select, len(b_to_select), replace = True),
                                            np.random.choice(s_to_select, len(s_to_select), replace = True)),
                                            axis =0)

            for i in random_bidders:
                re_df = pd.concat([re_df,data[data.member_code==i]])
            re_df = pd.concat([re_df,fixed0])
            re_df = re_df.sort_values('price',ascending = False).reset_index()
            re_df['cum_sum'] = re_df['appliedamount'].cumsum()

            cut_off_index = re_df[re_df.cum_sum >= Q].first_valid_index()
            
            #calculate cuf off price
            if cut_off_index is None:
                p =P_market

            else:
                p = re_df.price.iloc[cut_off_index,]
                

            
            pclist.append(p)
            pc_sum = pc_sum + p
            
            #record the amount he can win in this fake auction
            qc_i =  qc_i + fixed0[fixed0.price >= p].appliedamount.sum()

            count =count + 1
        #create fake bids for regression
        price1=min(pclist)
        price2=max(pclist)
        range1=price2-price1
        
        fake=pd.DataFrame()
        fake['price']=np.random.choice(range(int(price1*(1999/range1)),int(price2*(2003/range1)),1),2000,replace=False)/(2000/range1)
        
        fake = fake.sort_values('price',ascending = False)
        
        
        fake['cnt']=0
        #count times won for every row
        for p in pclist:
            fake['cnt'] = fake.apply(lambda x: x.cnt+1 if x.price>= p else x.cnt, axis=1)
        #caculate prob as the target of the model
        fake['prob'] = fake.cnt.values/itera
        setfixed=fake.loc[:,['price','prob']]
        X=setfixed.iloc[:,0:1].values
        
        Y=setfixed.iloc[:, 1].values
        #code for Polynomial regeression. I tried two model, and logistic is better
        #polyreg=plnm(degree=2)
        #change into logistic reg
        #Xp=polyreg.fit_transform(X)
        lreg=lrg()
        #logestic regression for prob
        Y1=np.where(Y>0.999, 0.999,Y) 
        Y1=np.where(Y1<0.001, 0.001, Y1) 
        Y2=np.log(Y1/(1-Y1))
        lreg=lrg()
        
        lreg.fit(X,Y2)
        plt.scatter(X,Y)
        plt.plot(X,deprob(lreg.predict(X)))
        plt.show()
        fixed0['amount_win_average'] = qc_i/itera
        
        
        
        fixed0=fixed0.sort_values('price',ascending = False)

        fixed0['real_cutoff'] = pc_sum/itera
        
        
            
        prob = deprob(lreg.predict(fixed0.iloc[:,4:5].values))
    
        fixed0['prob_win']=0
        fixed0['prob_win'].iloc[:,] = prob


        df = pd.concat([fixed0,df])

        count = 0
    df=df.reset_index(drop=True)
    return df
    

def deprob(y):
    x=np.exp(y)/(np.exp(y)+1)
    return x




#select the auctions you want to calculate 
auctions_all = df_raw.issueid.drop_duplicates().sort_values().tolist()
first = 7798
last  = 7799

index_f = bisect(auctions_all,first)
index_l = bisect(auctions_all,last)

auctions_selected = auctions_all[index_f-1:index_l]



#run the codes for the auctions selected
for auction in auctions_selected:

    try:
        df = fakebidder(df_raw,5,auction)  #change the number to change iteration time
        df.to_csv(str(auction)+'.csv')
    except:
        continue


#sum all the data into one csv
df_sup = pd.DataFrame()
for auction in auctions_selected:
    df1 = pd.read_csv(str(auction)+'.csv',index_col=False)
    df1 = df1.drop([df1.columns[0]],axis=1)
    df_sup = pd.concat([df_sup,df1])
df_sup.shape





df_sup.to_csv('final.csv')







