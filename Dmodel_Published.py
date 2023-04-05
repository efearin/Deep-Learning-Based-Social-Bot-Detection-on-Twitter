import torch.nn as nn
import torch
import torch.nn.functional as F

class DM (nn.Module):
    def __init__(self,inpsize:int=50,hiddensize:int=30,numberoflayers:int=1,
                 numberoftweets:int=20,tweetmetasize:int=16,accountmetasize:int=31,cnndepth:int=2,
                 bothsidelstm:int=1,samemodelfornontweet:int=1,partiallyclose:list=[]):

        super(DM,self).__init__()

        #TODO: hard coded - vector of dot in used word embeddings, change depending on your embeddings
        self.dot= torch.Tensor(  [ 0.68661 , -1.0772   , 0.011114 ,-0.24075 , -0.3422  ,  0.64456 ,  0.54957,
                           0.30411 , -0.54682 ,  1.4695   , 0.43648,  -0.34223 , -2.7189   , 0.46021,
                           0.016881 , 0.13953  , 0.020913,  0.050963, -0.48108 , -1.0764,   -0.16807,
                           -0.014315, -0.55055 ,  0.67823 ,  0.24359  ,-1.3179 ,  -0.036348 ,-0.228,
                           1.0337,   -0.53221 , -0.52934,   0.35537,  -0.44911 ,  0.79506,   0.56947,
                           0.071642, -0.27455,  -0.056911 ,-0.42961,  -0.64412 , -1.3495  ,  0.23258,
                           0.25383  ,-0.10226  , 0.65824 ,  0.16015  , 0.20959 , -0.067516, -0.51952,
                           -0.34922 ])

        self.partiallyclose=partiallyclose
        # list of the model parts that will be disabled
        # list can include: 'account', 'tweet', 'tweetmeta', 'description'

        self.bias=True

        self.bothsidelstm=bothsidelstm
        self.samemodelfornontweet=samemodelfornontweet

        self.dropoutprob = 0.3

        self.numberoftweets=numberoftweets
        self.inpsize=inpsize
        self.numberoflayers=numberoflayers
        self.hiddensize=hiddensize
        self.tweetmetasize=tweetmetasize

        #number of layers multiplier for description and tweet for both lstm case
        self.numberoflayersmul=1
        self.numberoflayersdescriptionmul=1
        self.hiddensizemul=1 #hiddensize multiplier for description and tweet for both lstm case

        self.dropout=nn.Dropout(self.dropoutprob)

        self.lstms=nn.ModuleList()
        for x in range(numberoftweets):
            torch.manual_seed(1) #needed to make initial weight values of all lstms same
            self.lstms.append(nn.LSTM(input_size=inpsize,hidden_size=hiddensize,num_layers=numberoflayers,batch_first =True,dropout=self.dropoutprob,bias=self.bias))

        if not bothsidelstm:
            self.inpshapetocnn =(accountmetasize+hiddensize,numberoftweets)
            self.cnnchannels=[1] +[(x+1)*numberoftweets for x in range(cnndepth) ]
            self.kernelsize=(7,3)
            self.padsize=[1,1,3,3] #for same padding for asymmetric kernel (7,3)

            self.nonlinear=nn.ReLU()
            self.pool = nn.MaxPool2d((2,2),2)
            self.dropout2d = nn.Dropout2d(self.dropoutprob)

            self.cnns=nn.ModuleList()
            for x in range(cnndepth):
                self.cnns.append(nn.Conv2d(in_channels=self.cnnchannels[x],out_channels=self.cnnchannels[x+1],kernel_size=self.kernelsize,stride=(1,1),bias=self.bias))

        else: #both side lstm
            self.twtlstm= nn.LSTM(input_size=(hiddensize+tweetmetasize),hidden_size=(int(self.hiddensizemul*hiddensize)),num_layers=self.numberoflayersmul*numberoflayers,batch_first =True,dropout=self.dropoutprob,bias=self.bias)

        #account description lstm
        self.descriptionlstm = nn.LSTM(input_size=inpsize,hidden_size=int(self.hiddensizemul*hiddensize),num_layers=self.numberoflayersdescriptionmul*self.numberoflayers,batch_first =True,dropout=self.dropoutprob,bias=self.bias)

        #TODO: update infeature by checking todo inside forward
        self.fc1=nn.Linear(in_features=2*int(self.hiddensizemul*hiddensize)+accountmetasize,out_features=1,bias=self.bias) #92
        #self.fc2=nn.Linear(in_features=20,out_features=1)
        self.out = nn.Sigmoid()

        if not samemodelfornontweet:
            self.fcnontweet=nn.Linear(in_features=accountmetasize+int(hiddensize*self.hiddensizemul), out_features=1,bias=self.bias)

    def getdescriptionout(self,accountdescription):
        if 'description' in self.partiallyclose: #description kapatılmışsa şayet
            descriptionout=torch.zeros(1,self.descriptionlstm.hidden_size).to(accountdescription.device)
        else:
            inithidden =(torch.zeros(self.numberoflayersdescriptionmul, 1, int(self.hiddensizemul*self.hiddensize)).to(accountdescription.device), torch.zeros(self.numberoflayersdescriptionmul, 1, int(self.hiddensizemul*self.hiddensize)).to(accountdescription.device))
            descriptionout, _ = self.descriptionlstm(accountdescription.unsqueeze(0), inithidden)
            descriptionout=descriptionout[:, -1, :]
        return descriptionout

    def getlstmout(self,tweets,tweetsmeta):
        availablenumoftweets=len(tweets)
        inithidden =(torch.zeros(1, 1, self.hiddensize).to(tweets[0].device), torch.zeros(1, 1, self.hiddensize).to(tweets[0].device))

        lstmout=[]

        for idx in range(availablenumoftweets): #tweetler yukarıda tek tek lstme veriliyor tweet metaları sonlarına ekleniyor
            if 'tweet' in self.partiallyclose: # tweetler kapalı
                lstmout.append(self.dropout(torch.cat([torch.zeros(self.hiddensize).to(tweets[0].device),tweetsmeta[idx].flatten()])))
            else:
                lstm=self.lstms[idx]
                out, _ = lstm(tweets[idx].unsqueeze(0), inithidden)
                if 'tweetmeta' in self.partiallyclose: # tweetmetalar kapalı
                    lstmout.append(self.dropout(torch.cat([out[:, -1, :].flatten(),torch.zeros(self.tweetmetasize).to(tweets[0].device)])))
                else: # tweet ve tweetmeta açık
                    lstmout.append(self.dropout(torch.cat((out[:, -1, :].flatten(),tweetsmeta[idx].flatten()))))
        return lstmout

    def getcnntwtout(self,lstmout):
        cnnout=torch.stack(lstmout, dim=1) #yeteri kadar twt yoksa 0 ile padle sabit img sizea gelsin
        cnnout = F.pad(cnnout,[0,self.inpshapetocnn[1]-cnnout.shape[1],0,0])
        cnnout=cnnout.unsqueeze(0).unsqueeze(0) #lstmout is lstm output of a tweet + tweet metadata in a column alt alta
        #cnne loop içinde verebilmek için 1,1 ekledik başına

        for idx,cnn in enumerate(self.cnns):
            cnnout=F.pad(cnnout,self.padsize)
            cnnout=cnn(cnnout)
            cnnout=self.nonlinear(cnnout)
            cnnout=self.pool(cnnout)
            cnnout=self.dropout2d(cnnout)
        twtout=cnnout
        return twtout

    def getlstmtwtout(self,lstmout):
        lstmout=lstmout[::-1] #ilk atılan tweetten son atılana doğru gitsin diye
        inittwthidden =(torch.zeros(self.numberoflayersmul, 1, int(self.hiddensizemul*self.hiddensize)).to(lstmout[0].device), torch.zeros(self.numberoflayersmul, 1, int(self.hiddensizemul*self.hiddensize)).to(lstmout[0].device))
        twtout, _ = self.twtlstm(torch.stack(lstmout).unsqueeze(0), inittwthidden)
        twtout=twtout[:, -1, :]
        return twtout

    def forward(self,tweets,tweetsmeta,accountdescription,accountmeta,getlastlayer=0):
        """

        :param tweets: list of tweets of an account - each element (tweet) is torch.tensor in shape [word size, embedding size]
        :param tweetsmeta: torch.tensor in shape [len(tweets), num of meta for each tweet]
        :param accountdescription: torch.tensor in shape [num of words in description, word embedding size]
        :param accountmeta: torch.tensor type element flat size like (Size)30
        :param getlastlayer:
        :return:
        """
        #get last layer 1 ise en son proba ya ulaşmadan önceki layer out ediliyor
        descriptionout=self.getdescriptionout(accountdescription)

        availablenumoftweets=len(tweets)

        if availablenumoftweets>0: #tweet inp verilmiş

            if 'tweet' in self.partiallyclose and 'tweetmeta' in self.partiallyclose: #tweetler ve tweetmetalar komple kapatılmışsa şayet
                twtout=torch.zeros(1,self.twtlstm.hidden_size).to(tweets[0].device)
            else: # tweetler ya da tweetmetalar açık
                lstmout= self.getlstmout(tweets,tweetsmeta)

                if not self.bothsidelstm:  #tweet lstmlerin çıktıları img haline getirilip cnne veriliyor
                    twtout=self.getcnntwtout(lstmout)

                else: #tweetler tek tek lstme veriliyor yukarıda sonra çıktıları da başka lstme veriliyor
                    twtout=self.getlstmtwtout(lstmout)

            if 'account' in self.partiallyclose:
                accountmeta=torch.zeros(accountmeta.shape).to(accountmeta.device)

            #TODO: update may need
            # open below 2 print first is to update fc in init second is to update to do one below in code
            #print(torch.cat((twtout.flatten(),accountmeta.flatten(),descriptionout.flatten())).shape)
            #print(twtout.flatten().shape)
            if getlastlayer:
                return torch.cat((twtout.flatten(),accountmeta.flatten(),descriptionout.flatten()))
            fcout=self.fc1(self.dropout(torch.cat((twtout.flatten(),accountmeta.flatten(),descriptionout.flatten()))))
            #fcout=self.fc2(fcout)


        else : #tweet yok
            if 'account' in self.partiallyclose: #burada twt zaten yok sadece account kapalı mı ona bak yeter
                accountmeta=torch.zeros(accountmeta.shape).to(accountmeta.device)
            if self.samemodelfornontweet:
                if getlastlayer:
                    return torch.cat((torch.zeros(30).to(accountmeta.device),accountmeta.flatten(),descriptionout.flatten()))
                #TODO update using one above to do print
                fcout=self.fc1(self.dropout(torch.cat((torch.zeros(30).to(accountmeta.device),accountmeta.flatten(),descriptionout.flatten()))))
            else:
                if getlastlayer:
                    return torch.cat([accountmeta.flatten(),descriptionout.flatten()])
                fcout=self.fcnontweet(torch.cat([accountmeta.flatten(),descriptionout.flatten()]))

        return self.out(fcout)


    def updategradsoflstms(self,partiallytrain,numberoftweets):
        #partiallytrain 1 ise sadece twt lstm gradlarını 0la
        if partiallytrain==0: #lstmler de update alacak avglerini al gradların
            for x in range(len(self.lstms[0].all_weights[0])):
                newgrad=0
                for y in range(numberoftweets):     #sadece kullanılanları avg yap #range(len(model.lstms)):
                    newgrad+=self.lstms[y].all_weights[0][x].grad
                newgrad=newgrad/numberoftweets      #newgrad/len(model.lstms)
                for y in range(len(self.lstms)):   #hepsini update et
                    self.lstms[y].all_weights[0][x].grad=newgrad
        else: #lstmler update almıcak 0 yap tüm gradları
            self.lstms.zero_grad()
        return self

def fillformissing(model,tweets,tweetsmeta,accountdescription, accountmeta):
    """
    fill missing oned vith vector of dot
    if no tweet exists single dot is pretended as be used and 0 vector used as tweetmeta
    :param model:
    :param tweets:
    :param tweetsmeta:
    :param accountdescription:
    :param accountmeta:
    :return:
    """
    if len(tweets)==0:
        tweets.append(model.dot.unsqueeze(0).to(accountmeta.device))
        tweetsmeta=torch.zeros(1,model.tweetmetasize).to(accountmeta.device)
    #temizlenmeden dolayı arada silinmiş tweetler varsa onların yerine de dot ın vektörünü koy
    for idx in range(len(tweets)):
        if len(tweets[idx])==0:
            tweets[idx]=model.dot.unsqueeze(0).to(accountmeta.device)
    #description yoksa onun yerine de dot vektörünü koy
    if len(accountdescription)==0:
        accountdescription = model.dot.unsqueeze(0).to(accountmeta.device)
    return model,tweets,tweetsmeta,accountdescription, accountmeta

def trainforanaccount(optimizer, criterion, model,tweets,tweetsmeta,accountdescription, accountmeta,accounttag, partiallytrain):

    model,tweets,tweetsmeta,accountdescription, accountmeta = fillformissing(model,tweets,tweetsmeta,accountdescription, accountmeta)

    optimizer.zero_grad() #necessary to clean previously remaning grads
    accounttagpred= model.forward(tweets,tweetsmeta,accountdescription,accountmeta) #make a prediction for account
    loss=criterion(accounttagpred,accounttag.flatten()) #calculate the loss with pred and real
    loss.backward()#get back comp. graph
    model = model.updategradsoflstms(partiallytrain,len(tweets)) if ((len(tweets)>0) and not('tweet' in model.partiallyclose)) else model #update grads of lstm as an average of all lstm grads in model, tweet yoksa hiç zaten 0 gradlar gerek yok buna
    optimizer.step()#use grads and update parameters
    return model, optimizer


