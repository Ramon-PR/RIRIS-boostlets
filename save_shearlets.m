%% SAVE shearlets
close all; clear; clc;

M = 512;
N = 128;

for M = [100, 128, 512, 1024]
    for N = [100, 128]
        sls_en = shearletSystem('ImageSize', [M, N], "PreserveEnergy", true, 'FilterBoundary', 'truncated');
        Psi = filterbank(sls_en);
        
        % plot_dicts(Psi, 5)
        check_dict(Psi)
        folder = "./ss_saved_dicts";
        save_dict(folder, Psi);
        fprintf("-----------------------------------\n")

    end
end


function save_dict(folder, Psi)
    [M, N, P] = size(Psi);
    filename = sprintf("SS_m_%i_n_%i.mat", M, N);
    
    if ~exist(folder, 'dir')
        mkdir(folder);
    end

    % Define la ruta completa al archivo
    filepath = fullfile(folder, filename);
    
    % Guarda la variable Sk en el archivo
    save(filepath, 'Psi');
    
    fprintf("\t ShearletSystem Psi[%i, %i, %i] saved \n", M, N, P ); 
end


function [] = plot_dicts(Sk, num_cols)
    [m,n,p] = size(Sk);
    fprintf("\t size Sk = [M,N,P] = [%i, %i, %i] \n", m, n, p)
    
    num_rows = ceil(p/num_cols); 
    
    figure;
    t = tiledlayout(num_rows, num_cols, 'Padding', 'none', 'TileSpacing','Compact');
    for i=1:p
        Sk2 = Sk(:,:,i).*Sk(:,:,i);
        nexttile;
        pcolor(Sk2); shading interp;
        title(['i = ', num2str(i)], Interpreter='latex');
        set(gca,'XTick',[], 'YTick', []);
    end
end

function [] = check_dict(Sk)
    
    sum_sk2 = sum(Sk.*Sk, 3);
    max_sk2 = max(sum_sk2, [], "all");
    min_sk2 = min(sum_sk2, [], "all");
    fprintf("\t min(sum(Sk*Sk)) = %2.1f,   max(sum(Sk*Sk)) = %2.1f \n", ...
        min_sk2, max_sk2);

end


