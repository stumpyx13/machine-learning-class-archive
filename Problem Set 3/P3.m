%6.867 PSET 3 Problem 3 - Latent Dirichlet Allocation with 20newsgroups data
%% Options
createBag                           =   0;
fitmodel                            =   1;
plotWordCloud                       =   0;
modifyBag                           =   1;
plotRandomTopics                    =   1;
randomTopicPlotCount                =   10;

dataPath                            =   'hw3_resources/20newsgroups/*';
N_topics                            =   100;

%% Article numbers of interest
%rec.auto
N_article1                          =   [7058, 8061];

%sci.space
N_article2                          =   [14070, 15070];

%% Load data into filedatastore object in MATLAB
fds                                 =   fileDatastore(dataPath,'ReadFcn',@extractFileText);
fds.Files                           =   fds.Files(2:end);

N_files                             =   length(fds.Files);
%% Read in data to a bag of words model
if(createBag)
    bag                                 =   bagOfWords;
    N_fileread                          =   0;
    while hasdata(fds)
        N_fileread                      =   N_fileread + 1;
        disp(['Reading file ',num2str(N_fileread),' of ',num2str(N_files),'..']);
        str                             =   read(fds);
        document                        =   tokenizedDocument(str);
        bag                             =   addDocument(bag,document);
    end
end

%% modify bag of words
if(modifyBag)
    bag                                     =   removeWords(bag,stopWords);
    bag                                     =   removeShortWords(bag,1);
    bag                                     =   removeWords(bag,["The","In","to","Is","I'm","It","edu","com",...
                                                    "If","And","This","You","like","So","They","EDU","cc",...
                                                    "org","ca","Edu","got","That","CA","just","get","say",...
                                                    "I'd","For","There","That","But","I've","Up","Thanks",...
                                                    "We","COM","As", "Article","Writes","article","writes",...
                                                    "He"]);
end

%% Fit model
if(fitmodel)
    mdl                                 =   fitlda(bag,N_topics);
end

%% plot word cloud
if(plotWordCloud)
    f                                       =   figure(1);
    set(f,'Position',[200,200, 2000, 800]);
    for topicIdx = 1:N_topics
        subplot(2,N_topics/2,topicIdx)
        wordcloud(mdl,topicIdx);
        title("Topic: " + topicIdx)
    end
end

if(plotRandomTopics)
   h                                    =   figure(2);
   set(h,'Position',[200,200, 2000, 800]);
   rr                                   =   randi(N_topics,1,randomTopicPlotCount);
   for topicIdx = 1:length(rr)
       subplot(2,randomTopicPlotCount/2,topicIdx)
       wordcloud(mdl,rr(topicIdx));
       title("Topic: " + rr(topicIdx));
   end
end

%% Find topics associated with article class 1 and 2
article1_topicProb                          =   mdl.DocumentTopicProbabilities(N_article1(1):N_article1(2),:);
article2_topicProb                          =   mdl.DocumentTopicProbabilities(N_article2(1):N_article2(2),:);

article1_topicProb_avg                      =   mean(article1_topicProb,1);
article2_topicProb_avg                      =   mean(article2_topicProb,1);

[val1, ind1]                                =   sort(article1_topicProb_avg,'descend');
[val2, ind2]                                =   sort(article2_topicProb_avg,'descend');

top3Topics_art1                             =   ind1(1:3);
top3Topics_art2                             =   ind2(1:3);

ff                                          =   figure(3);
set(h,'Position',[200,200, 2000, 800]);
for i = 1:3
    subplot(2,3,i)
    wordcloud(mdl,top3Topics_art1(i))
    title("Topic: " + top3Topics_art1(i));
end
for i = 1:3
    subplot(2,3,i+3)
    wordcloud(mdl,top3Topics_art2(i))
    title("Topic: " + top3Topics_art2(i));
end
