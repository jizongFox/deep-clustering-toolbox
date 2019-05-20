function [S] = clusterPGD(X,params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%%%%%% Projected gradient descent method to compute self-representation matrix S

% Intialization and pre-computed values
S = normrnd(0,0.001,size(X,2));
I = eye(size(S));

X2 = X'*X;


diff = [];
loss = [];

% Main loop
for t=1:params.maxIter    
    Z = S - params.learnRate*X2*(S - I);      
    
    if params.lowRank
        % Low rank prior
        [L,D,R] = svd(Z,'econ');
        DD = max(D-(params.lambda*params.learnRate),0);
        Snew = L*DD*R';      
        Snew(1:1+size(Snew,1):end) = 0; 
        reg = sum(diag(DD));
    else
        % Sparse prior        
        Snew = sign(Z).*max(abs(Z)-(params.lambda*params.learnRate),0);    
        Snew(1:1+size(Snew,1):end) = 0; 
        % Set diagonal elements to 0
        reg = sum(sum(abs(Snew)));
    end
            
    diff(end+1) = norm(S-Snew,'fro')/norm(S,'fro');
    
    S = Snew;
    
    % loss
    loss(end+1) = 0.5*(norm(X - X*S, 'fro').^2) + params.lambda*reg;   
    
    if mod(t,params.printInfo) == 0
        figure(2), imagesc(S), caxis([0 1]), title('PGD clustering');%
        %figure(2), imagesc(Z);%spy(S);
        colorbar;
        figure(4), plot(1:t,diff), title('PGD convergence');
        drawnow;
        fprintf('Iter %d\t: diff=%f,\t loss=%f\n', t, diff(end), loss(end));
    end
    
    if diff(end) < params.minDiff
        % Converged
        break;
    end
end

fprintf('Final iter\t: diff=%f,\t loss=%f\n', diff(end), loss(end));

end

