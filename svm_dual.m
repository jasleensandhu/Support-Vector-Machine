tic;
trainingset = csvread('C:/UTA/Fall2016/Machine Learning/spam-20161006T192251Z/spam/spambase_train.data',0,0,[0 0 2759 56]);
Y = csvread('C:/UTA/Fall2016/Machine Learning/spam-20161006T192251Z/spam/spambase_train.data',0,57);
Y(Y==0)=-1;
numOfExamples = 2760;
numOfAttributes=57;
sigma=1;

% Making the kernal matrix
kernel = zeros(numOfExamples,numOfExamples);
for i=1:numOfExamples
    for j=1:numOfExamples
        kernel(i,j)= gaussianKernel(trainingset(i,:)',trainingset(j,:)',sigma);
        
    end;
end;
H=Y*Y'*kernel;
C=1;

f=-ones(1,numOfExamples);

Aeq=Y';
beq=0;
LB=zeros(numOfExamples,1);
UB=C*ones(numOfExamples,1);
alpha=quadprog(H,f,[],[],Aeq,beq,LB,UB);
temp=zeros(numOfExamples,1);

% calculating w and b from alphas
summation=diag(Y)*alpha;
for i=1:numOfExamples
   
    temp(i,1)=summation'*kernel(i,:)';
end

w=summation'*trainingset;

bm=Y-temp;
b=mean(bm);


%Testing the data
testingdata = csvread('C:\UTA\Fall2016\Machine Learning\spam-20161006T192251Z\spam\spambase_validation.data',0,0,[0,0,919,56]);
labels=csvread('C:\UTA\Fall2016\Machine Learning\spam-20161006T192251Z\spam\spambase_validation.data',0,57);

labels(labels==0)=-1; 

testLabels= [testingdata ones(920,1)]*[w b]';
temp=diag(labels)*testLabels;
misclassifications=temp(temp<0);
s =size(misclassifications,1);
accuracy=(920-s)*100/920;
display(accuracy);
TimeSpent = toc;
display(TimeSpent,'Time for execution (in seconds)')