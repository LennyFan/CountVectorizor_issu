

# if binary is True, it will count only once if the word is in the string
test_sender = CountVectorizer(analyzer = "word",stop_words='english' ,token_pattern='[A-Za-z]+(?=\\s+)',binary = True
                              , vocabulary=['bill']
                             )
fit = test_sender.fit_transform(['the bill com team notification only hq bill ', 'receipt ', ' bill '])
print(fit.toarray())#.sum(axis=0))
#test_sender.get_feature_names()
#matrix = count_vect.fit_transform(doc_list)
freqs_ = [(word, fit.getcol(idx).sum()) for word, idx in test_sender.vocabulary_.items()]
test_sender.vocabulary_.items()
fit.getcol(0)#.sum()

>>> [[0][0][0]]

freqs_

>>> [('bill', 0)]



# if binary is True, it will count only once if the word is in the string
test_sender = CountVectorizer(analyzer = "word",stop_words='english' ,token_pattern='[A-Za-z]+(?=\\s+)',binary = True
                             )
fit = test_sender.fit_transform(['the bill com team notification only hq bill ', 'receipt ', ' bill '])
print(fit.toarray())#.sum(axis=0))
#test_sender.get_feature_names()
#matrix = count_vect.fit_transform(doc_list)
freqs_ = [(word, fit.getcol(idx).sum()) for word, idx in test_sender.vocabulary_.items()]
test_sender.vocabulary_.items()
fit.getcol(0)#.sum()


>>> [[1 1 1 0 1]
     [0 0 0 1 0]
     [0 0 0 0 0]]
     
freqs_

>>> [('com', 1), ('team', 1), ('notification', 1), ('hq', 1), ('receipt', 1)]
     
     
