import os
import requests
import json
import pandas as pd
from pandas import read_csv
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from datetime import datetime
import pathlib

script_path = pathlib.Path(__file__).parent.absolute()

#### Paths
DATAPATH = os.path.join(script_path,'data/')
RESULTPATH = os.path.join(script_path,'results/')
MODELPATH = os.path.join(script_path,'models/')
PREDICTPATH = os.path.join(script_path,'predictions/')
topicsfile = os.path.join(DATAPATH,'litcovidtopics.tsv')
topicsdf = read_csv(topicsfile,delimiter='\t',header=0,index_col=0)
topiclist = topicsdf['topicCategory'].unique().tolist()


#### Fetch Relevant metadata
def batch_fetch_meta(idlist):
    ## Break the list of ids into smaller chunks so the API doesn't fail the post request
    runs = round((len(idlist))/100,0)
    i=0 
    separator = ','
    ## Create dummy dataframe to store the meta data
    textdf = pd.DataFrame(columns = ['_id','abstract','name'])
    while i < runs+1:
        if len(idlist)<100:
            sample = idlist
        elif i == 0:
            sample = idlist[i:(i+1)*100]
        elif i == runs:
            sample = idlist[i*100:len(idlist)]
        else:
            sample = idlist[i*100:(i+1)*100]
        sample_ids = separator.join(sample)
        ## Get the text-based metadata (abstract, title) and save it
        r = requests.post("https://api.outbreak.info/resources/query/", params = {'q': sample_ids, 'scopes': '_id', 'fields': 'name,abstract'})
        if r.status_code == 200:
            rawresult = pd.read_json(r.text)
            cleanresult = rawresult[['_id','name','abstract',]].loc[rawresult['_score']==1].copy()
            cleanresult.drop_duplicates(subset='_id',keep="first", inplace=True)
            textdf = pd.concat((textdf,cleanresult))
        i=i+1
    return(textdf)
        
    
    
#### Transform metadata
def merge_texts(df):
    df.fillna('',inplace=True)
    df['text'] = df['name'].astype(str).str.cat(df['abstract'],sep=' ')
    df['text'] = df['text'].str.replace(r'\W', ' ')
    df['text'] = df['text'].str.replace(r'\s+[a-zA-Z]\s+', ' ')
    df['text'] = df['text'].str.replace(r'\^[a-zA-Z]\s+', ' ')
    df['text'] = df['text'].str.lower()   
    return(df)


def fetch_categorized_data(df):
    alldata = pd.DataFrame(columns=['_id','name','abstract','text','topicCategory'])
    breakdown = df.groupby('topicCategory').size().reset_index(name='counts')
    for eachtopic in breakdown['topicCategory'].tolist():
        tmpids = df['_id'].loc[df['topicCategory']==eachtopic]
        tmptxtdf = batch_fetch_meta(tmpids)
        tmptxtdf = merge_texts(tmptxtdf)
        tmptxtdf['topicCategory']=eachtopic
        alldata = pd.concat((alldata,tmptxtdf),ignore_index=True)
    return(alldata)


def generate_training_df(df,category):
    positiveids = df['_id'].loc[df['topicCategory']==category].tolist()
    training_set_pos = df[['_id','text']].loc[df['topicCategory']==category].copy()
    training_set_pos['target']='in category'
    max_negs = len(positiveids)
    if len(positiveids)<len(df.loc[~df['_id'].isin(positiveids)]):
        training_set_neg = df[['_id','text']].loc[~df['_id'].isin(positiveids)].sample(n=max_negs).copy()
    else:
        training_set_neg = df[['_id','text']].loc[~df['_id'].isin(positiveids)].copy()
    training_set_neg['target']='not in category'
    training_set = pd.concat((training_set_pos,training_set_neg),ignore_index=True)
    return(training_set)


#### Note that this function is to clean up the classification predictions and format it as annotations
def clean_results(allresults):
    counts = allresults.groupby('_id').size().reset_index(name='counts')
    duplicates = counts.loc[counts['counts']>1]
    singles = counts.loc[counts['counts']==1]
    dupids = duplicates['_id'].unique().tolist()
    tmplist = []
    for eachid in dupids:
        catlist = allresults['topicCategory'].loc[allresults['_id']==eachid].tolist()
        tmplist.append({'_id':eachid,'topicCategory':catlist})
    tmpdf = pd.DataFrame(tmplist)  
    tmpsingledf = allresults[['_id','topicCategory']].loc[allresults['_id'].isin(singles['_id'].tolist())]
    idlist = tmpsingledf['_id'].tolist()
    catlist = tmpsingledf['topicCategory'].tolist()
    cattycat = [[x] for x in catlist]
    list_of_tuples = list(zip(idlist,cattycat))
    singledf = pd.DataFrame(list_of_tuples, columns = ['_id', 'topicCategory']) 
    cleanresults = pd.concat((tmpdf,singledf),ignore_index=True)
    return(cleanresults)



#### Train classifiers
def train_test_classify(classifier,training_set,X,i=0):
    X_train, X_test, y_train, y_test = train_test_split(X, training_set.target, test_size=0.2, random_state=i)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cmresult = confusion_matrix(y_test,y_pred)
    report = pd.DataFrame(classification_report(y_test,y_pred,output_dict=True))
    probs = classifier.predict_proba(X_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    return(cmresult,report,auc)


def vectorize_text(training_set):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(training_set['text'])
    features = vectorizer.get_feature_names()
    return(X)


def generate_vectorizer(training_set,category):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(training_set['text'])
    features = vectorizer.get_feature_names()
    vectorizerfile = os.path.join(MODELPATH,"vectorizer_"+category+".pickle")
    xfile = os.path.join(MODELPATH,"X_"+category+".pickle")
    pickle.dump(vectorizer, open(vectorizerfile, "wb"))
    pickle.dump(X, open(xfile, "wb"))
    return(X)


def save_model(classifier,classname,category):
    filename = os.path.join(MODELPATH,classname+"_"+category+".sav")
    pickle.dump(classifier, open(filename, 'wb'))

    
def load_classifiers(classifierset_type):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neural_network import MLPClassifier
    from sklearn import tree
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    all_available = {
        'Random Forest':RandomForestClassifier(n_estimators=1000, random_state=0),
        'MultinomialNB':MultinomialNB(),
        'Neural Net':MLPClassifier(alpha=1, max_iter=1000),
        'Decision Tree':tree.DecisionTreeClassifier(max_depth=5),
        'Nearest Neighbor':KNeighborsClassifier(3),
        'AdaBoost':AdaBoostClassifier(),
        'Logistic Regression':LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')}
    best = {
        'Random Forest':RandomForestClassifier(n_estimators=1000, random_state=0),
        'MultinomialNB':MultinomialNB(),
        'Logistic Regression':LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')}
    if classifierset_type=='best':
        return(best)
    else:
        return(all_available)
    
    
    
    
###### Fetch Preprint data from our outbreak API
#### Get the size of the source (to make it easy to figure out when to stop scrolling)
def fetch_src_size(source):
    pubmeta = requests.get("https://api.outbreak.info/resources/query?q=curatedBy.name:"+source+"&size=0&aggs=@type")
    pubjson = json.loads(pubmeta.text)
    pubcount = int(pubjson["facets"]["@type"]["total"])
    return(pubcount)

#### Pull ids from a json file
def get_ids_from_json(jsonfile):
    idlist = []
    for eachhit in jsonfile["hits"]:
        if eachhit["_id"] not in idlist:
            idlist.append(eachhit["_id"])
    return(idlist)

#### Ping the API and get all the ids for a specific source and scroll through the source until number of ids matches meta
def get_source_ids(source):
    source_size = fetch_src_size(source)
    r = requests.get("https://api.outbreak.info/resources/query?q=curatedBy.name:"+source+"&fields=_id&fetch_all=true")
    response = json.loads(r.text)
    idlist = get_ids_from_json(response)
    try:
        scroll_id = response["_scroll_id"]
        while len(idlist) < source_size:
            r2 = requests.get("https://api.outbreak.info/resources/query?q=curatedBy.name:"+source+"&fields=_id&fetch_all=true&scroll_id="+scroll_id)
            response2 = json.loads(r2.text)
            idlist2 = set(get_ids_from_json(response2))
            tmpset = set(idlist)
            idlist = tmpset.union(idlist2)
            try:
                scroll_id = response2["_scroll_id"]
            except:
                print("no new scroll id")
        return(idlist)
    except:
        return(idlist)

#### Pull ids from the major publication sources (litcovid, medrxiv,biorxiv)
def get_preprint_ids():
    biorxiv_ids = get_source_ids("bioRxiv")
    medrxiv_ids = get_source_ids("medRxiv")
    preprint_ids = list(set(medrxiv_ids).union(set(biorxiv_ids)))
    return(preprint_ids)
    

    
    
#### Using Pre-trained models
def load_vectorizer(category):
    vectorizerfile = os.path.join(MODELPATH,"vectorizer_"+category+".pickle")
    vectorizer = pickle.load(open(vectorizerfile,'rb'))
    return(vectorizer)

def predict_class(topiclist,classifierlist,df):
    labels = df['_id']
    for eachtopic in topiclist:
        vectorizer = load_vectorizer(eachtopic)
        M = vectorizer.transform(df['text'])
        for eachclassifier in classifierlist:
            classifierfile = os.path.join(MODELPATH, eachclassifier+"_"+eachtopic+'.sav')
            classifier = pickle.load(open(classifierfile, 'rb'))
            prediction = classifier.predict(M)
            list_of_tuples = list(zip(labels,prediction))
            predictiondf = pd.DataFrame(list_of_tuples, columns = ['_id', 'prediction'])
            predictiondf['topicCategory']=eachtopic
            predictiondf['classifier']=eachclassifier
            predictiondf.to_csv(os.path.join(PREDICTPATH,eachtopic+"_"+eachclassifier+'.tsv'),sep='\t',header=True)    

            
            
            
            
            
#### Evauating Predictions
def get_agreement(eachtopic,classifierlist,PREDICTPATH):
    agreement = pd.DataFrame(columns=['_id','topicCategory','pos_pred_count','pos_pred_algorithms'])
    classresult = pd.DataFrame(columns=['_id','prediction','topicCategory','classifier'])
    for eachclass in classifierlist:
        tmpfile = read_csv(os.path.join(PREDICTPATH,eachtopic+"_"+eachclass+".tsv"),delimiter='\t',header=0,index_col=0)
        classresult = pd.concat((classresult,tmpfile),ignore_index=True)
    posresults = classresult.loc[classresult['prediction']=='in category']
    agreecounts = posresults.groupby('_id').size().reset_index(name='counts')
    no_agree = posresults.loc[posresults['_id'].isin(agreecounts['_id'].loc[agreecounts['counts']==1].tolist())].copy()
    no_agree.rename(columns={'classifier':'pos_pred_algorithms'},inplace=True)
    no_agree['pos_pred_count']=1
    no_agree.drop('prediction',axis=1,inplace=True)
    perfect_agree = posresults.loc[posresults['_id'].isin(agreecounts['_id'].loc[agreecounts['counts']==len(classifierlist)].tolist())].copy()
    perfect_agree['pos_pred_count']=len(classifierlist)
    perfect_agree['pos_pred_algorithms']=str(classifierlist)
    perfect_agree.drop(['prediction','classifier'],axis=1,inplace=True)
    perfect_agree.drop_duplicates('_id',keep='first',inplace=True)
    partialcountids = agreecounts['_id'].loc[((agreecounts['counts']>1)&
                                          (agreecounts['counts']<len(classifierlist)))].tolist()
    tmplist = []
    for eachid in list(set(partialcountids)):
        tmpdf = posresults.loc[posresults['_id']==eachid]
        tmpdict = {'_id':eachid,'topicCategory':eachtopic,'pos_pred_count':len(tmpdf),
                   'pos_pred_algorithms':str(tmpdf['classifier'].tolist())}
        tmplist.append(tmpdict)
    partial_agree = pd.DataFrame(tmplist)    
    agreement = pd.concat((agreement,no_agree,partial_agree,perfect_agree),ignore_index=True)
    return(agreement)

def filter_agreement(topiclist,classifierlist,agreetype='perfect'):
    allagreement = pd.DataFrame(columns=['_id','topicCategory','pos_pred_count','pos_pred_algorithms'])
    for eachtopic in topiclist:
        agreement = get_agreement(eachtopic,classifierlist,PREDICTPATH)
        allagreement = pd.concat((allagreement,agreement),ignore_index=True)
    if agreetype=='perfect':
        filtered_agreement = allagreement[['_id','topicCategory']].loc[allagreement['pos_pred_count']==len(classifierlist)].copy()
    elif agreetype=='None':
        filtered_agreement = allagreement[['_id','topicCategory']].loc[allagreement['pos_pred_count']==1].copy()
    else:
        partialcountids = allagreement['_id'].loc[((allagreement['pos_pred_count']>1)&
                                          (allagreement['pos_pred_count']<len(classifierlist)))].tolist()
        filtered_agreement = allagreement[['_id','topicCategory']].loc[allagreement['_id'].isin(partialcountids)].copy()
    return(filtered_agreement)


def merge_predictions(topiclist,classifierlist,agreetype='perfect'):
    totalagree = pd.DataFrame(columns=['_id','topicCategory'])
    for eachtopic in topiclist:
        agreement = filter_agreement(topiclist,classifierlist,agreetype='perfect')
        totalagree = pd.concat((totalagree,agreement),ignore_index=True)
    totalagree.drop_duplicates(inplace=True,keep="first")
    return(totalagree)




###### Primary Functions
def run_test(topicsdf,classifierset_type='best',export_report=False):
    classifiers = load_classifiers(classifierset_type)
    fetchstarttime = datetime.now()
    print("fetching the abstracts: ", fetchstarttime)
    alldata = fetch_categorized_data(topicsdf)
    fetchtime = datetime.now()-fetchstarttime
    print("fetching complete: ",fetchtime)
    breakdown = alldata.groupby('topicCategory').size().reset_index(name='counts')
    testresults = []
    for eachtopic in breakdown['topicCategory'].tolist():
        print("now testing: ",eachtopic,datetime.now())
        trainingset = generate_training_df(alldata,eachtopic)
        X = vectorize_text(trainingset)
        for classifier in classifiers.keys():
            i=0
            while i<5:
                timestart = datetime.now()
                cmresult,report,auc = train_test_classify(classifiers[classifier],training_set,X,i)
                runtime = datetime.now() - timestart
                testresults.append({'topicCategory':eachtopic,'set size':len(training_set),'classifier':classifier,
                                    'runtime':runtime,'auc':auc,'report':report,'matrix':cmresult,'i':i})
                i=i+1
    testresultsdf = pd.DataFrame(testresults)
    if export_report==True:
        testresultsdf.to_csv(os.path.join(RESULTPATH,'in_depth_classifier_test.tsv'),sep='\t',header=True)
    return(testresultsdf)


def generate_models(topicsdf,classifiers):
    alldata = fetch_categorized_data(topicsdf)
    breakdown = alldata.groupby('topicCategory').size().reset_index(name='counts')

    for eachtopic in breakdown['topicCategory'].tolist():
        trainingset = generate_training_df(alldata,eachtopic)
        X = generate_vectorizer(trainingset,eachtopic)
        for eachclassifier in classifiers.keys():
            classifier=classifiers[eachclassifier]
            classifier.fit(X, trainingset.target)
            save_model(classifier,eachclassifier,eachtopic)  

            
def classify_preprints(topiclist,classifiers):
    preprint_ids = get_preprint_ids()
    preprintdf = batch_fetch_meta(preprint_ids)
    preprintdata = merge_texts(preprintdf)    
    classifierlist = classifiers.keys()
    predict_class(topiclist,classifierlist,preprintdata)

    
def load_annotations(topiclist,classifiers):
    classify_preprints(topiclist,classifiers)
    classifierlist = classifiers.keys()
    total_agree = merge_predictions(topiclist,classifierlist,agreetype='perfect')
    cleanresults = clean_results(total_agree)
    cleanresults.to_json(os.path.join(RESULTPATH,'preprint_categories.json'),orient="records")
        


