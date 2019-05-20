clc
close all;

K = 10; % Number of clusters
N = 10; % Number of examples per cluster
M = 10; % Number of features per example
var = .1; % Variance of each cluster

% Generate random dat ausing K clusters
X = [];

for k=1:K
    mm = rand(M,1);
    for i=1:N
        X = [X normrnd(mm,var)];
    end
end

% Choose between sparse (lowRank=false) or low rank (lowRank=true) prior for S
params.lowRank = false;

% Sparse prior model:
% argmin_S (1/2)||X - XS||_F^2 + lambda*||S||_1, s.t. diag(S) = 0

% Low rank prior model:
% argmin_S (1/2)||X - XS||_F^2 + lambda*||S||_*
%    where ||o||_* is the nuclear norm (sum of singular values)

if params.lowRank
    params.lambda = 1;
else
    params.lambda = .1;
end

params.maxIter = 10000;

% ADMM parameters
params.mu = 1;
params.minRes = 1e-6;

% Projected gradient descent parameters
params.learnRate = .005;
params.minDiff = 1e-6;

% Number of iterations before showing matrix
params.printInfo = 1000000000;

% S1 = clusterADMM(X,params);
S2 = clusterPGD(X,params);
   
% figure(1), imagesc(S1), caxis([0 1]), title('ADMM');
% colorbar;
figure(2), imagesc(S2), caxis([0 1]), title('PGD');
%figure(2), imagesc(Z);%spy(S);
colorbar;

%%

%{
clc
close all
gamma = 10;
eps = 1e-1;

I = eye(size(S));
T = (I-S);
T = T*T';

W = S + S';
D = diag(1./sqrt(sum(W,1) + eps));
L = I - D*W*D;
LL = diag(sum(W,1) + eps) - W;

%figure(5), imagesc(TT)

%Xhat = X*inv(eye(size(S)) + gamma*TT);

%figure(5), imagesc(X);
%figure(6), imagesc(Xhat);

[U,D,V] = svds(T,3,'smallest');
U = U(:,1:2);

[U2,D2,V2] = svds(L,3,'smallest');
U2 = U2(:,1:2);


colors = [];

for k=1:K
    colors = [colors; k*ones(N,1)];
end


figure(10), scatter(U(:,1), U(:,2),10,colors)
figure(11), scatter(U2(:,1), U2(:,2),10,colors)
%}

