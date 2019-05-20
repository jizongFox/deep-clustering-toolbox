function [S] = clusterADMM(X,params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%%%%%% ADMM method to compute self-representation matrix S

% Intialization and pre-computed values
S = zeros(size(X,2));
U = zeros(size(S));
X2 = X'*X;
T = inv(X2 + params.mu*eye(size(X,2)));


res = [];
loss = [];

% Main loop
for t=1:params.maxIter    
    Z = T*(X2 + params.mu*(S+U));
    
    if params.lowRank
        % Low rank prior
        [L,D,R] = svd(Z-U,'econ');
        DD = max(D-(params.lambda/params.mu),0);
        S = L*DD*R';  
        S(1:1+size(S,1):end) = 0; 
        reg = sum(diag(DD));
    else
        % Sparse prior        
        S = sign(Z-U).*max(abs(Z-U)-(params.lambda/params.mu),0);   
        S(1:1+size(S,1):end) = 0;  
        reg = sum(sum(abs(S)));
        % Set diagonal elements to 0        
    end
         
    U = U + (S-Z);
    
    % Constraint residual
    res(end+1) = norm(S-Z, 'fro')/norm(Z, 'fro');
    
    % Loss        
    loss(end+1) = 0.5*(norm(X - X*S, 'fro').^2) + params.lambda*reg;
    
    if mod(t,params.printInfo) == 0
        figure(1), imagesc(S), caxis([0 1]), title('ADMM clustering');%
        %figure(2), imagesc(Z);%spy(S);
        colorbar;
        figure(3), plot(1:t,res), title('ADMM convergence');
        drawnow;
        fprintf('Iter %d\t: res=%f,\t loss=%f\n', t, res(end), loss(end));
    end
    
    if res(end) < params.minRes
        % Converged
        break;
    end
end

fprintf('Final iter\t: res=%f,\t loss=%f\n', res(end), loss(end));

end

