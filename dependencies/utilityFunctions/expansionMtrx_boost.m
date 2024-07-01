function [E,e] = expansionMtrx_boost(imSize, S, n_thetas)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPANSIONMTRX Expansion matrix restricting vertical-like shearlets
% IN    imSize: 2-element array with image dimensions (T,M)
%       tau:    no. decomposition scales
% -----------------------------------------------------------------------------------
% OUT   E:      expansion matrix (rows removed for vertical shearlets)
%       e:      vectorized diagonal of E
% -----------------------------------------------------------------------------------
% AUTHOR: E. Zea (zea@kth.se)
% DATE: 2019-03-05
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nc = imSize(1)*imSize(2);   % no. total image samples
% Ns = 1+sum(2.^([1:tau]+1)); % no. shearlets (Eq. 5)
Ns = 1 + 2*S*n_thetas; % scaling_fun & far_or_near*a_grid*theta_grid


% indices with vertical-like orientation
% switch S
%     case 1
%         vertShears = 2;
%     case 2
%         vertShears = [2, 6,7,13];
%     case 3
%         vertShears = [2, 6,7,13, 14,15,16,17,27,28,29];
%     case 4
%         vertShears = [2, 6,7,13, 14,15,16,17,27,28,29, 30,31:37,55:61];
%     case 5
%         vertShears = [2, 6,7,13, 14,15,16,17,27,28,29, 30,31:37,55:61, ...
%                       62,63:77,111:125];
% end

vertBoostlets = [];

% build expansion matrix
set = [];
e = ones(Ns*Nc,1); 
E = sparse(1:numel(e),1:numel(e),e); % sparse expansion matrix
for uu = 1:numel(vertBoostlets)
    set = [set; (vertBoostlets(uu)-1)*Nc+1:vertBoostlets(uu)*Nc ];
end
% remove columns corresponding to vertical shearlets
E(:,set) = []; 
end