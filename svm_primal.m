tic;
Training = csvread('C:/UTA/Fall2016/Machine Learning/spam-20161006T192251Z/spam/spambase_train.data',0,0,[0 0 2759 56]);
Group = csvread('C:/UTA/Fall2016/Machine Learning/spam-20161006T192251Z/spam/spambase_train.data',0,57);
Group(Group==0)=-1;
numOfExamples = 2760;
numOfAttributes=57;
C=100;
H = diag([ones(1, numOfAttributes), zeros(1, numOfExamples+1),0]);
f = [zeros(1, numOfAttributes), C * ones(1, numOfExamples), 0,0]';

Z = [Training ones(numOfExamples,1)];
A=-diag(Group)*Z ;


Aineq = [A, -1*eye(numOfExamples), zeros(numOfExamples,1)];
bineq = -1*ones(numOfExamples, 1);
LB = [-inf*ones(1,numOfAttributes+1),zeros(1,numOfExamples),-inf]';
UB = inf*ones(2819,1);
options = optimset('Algorithm','interior-point-convex');
Result=quadprog(H,f,Aineq,bineq,[],[],LB,UB,[],options);

wT = Result(1:58);
e= Result(59:end-1);

testdata = csvread('C:\UTA\Fall2016\Machine Learning\spam-20161006T192251Z\spam\spambase_validation.data',0,0,[0,0,919,56]);
labels=csvread('C:\UTA\Fall2016\Machine Learning\spam-20161006T192251Z\spam\spambase_validation.data',0,57);

labels(labels==0)=-1; 

testLabels= [testdata ones(920,1)]*wT;
temp=diag(labels)*testLabels;
misclassifications=temp(temp<0);
s =size(misclassifications,1);
accuracy=(920-s)*100/920;
display(accuracy);
TimeSpent = toc;
display(TimeSpent,'Time for execution (in seconds)');