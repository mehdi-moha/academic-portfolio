clc
clear
close all

filename = 'Data.csv';
delimiter = ',';
startRow = 3;
formatSpec = '%f%f%f%f%f%f%f%f%f%[^\n\r]';

fileID = fopen(filename, 'r');

dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines', startRow-1, 'ReturnOnError', false);

fclose(fileID);

Sample = dataArray{:, 1};
x1 = dataArray{:, 2};
x2 = dataArray{:, 3};
x3 = dataArray{:, 4};
x4 = dataArray{:, 5};
x5 = dataArray{:, 6};
x6 = dataArray{:, 7};
x7 = dataArray{:, 8};
x8 = dataArray{:, 9};

clearvars filename delimiter startRow formatSpec fileID ans;

Data = [ones(10, 1), x1, x2; ones(10, 1), x3, x4; ones(10, 1), x5, x6; ones(10, 1), x7, x8]';

tW = [];
for s = 1:4
    hold on;
    switch s
        case 1
            x = [Data(:, 1:10), Data(:, 11:end)];
            str = 'Class 1 vs Other Classes';
        case 2
            x = [Data(:, 11:20), Data(:, [1:10, 21:end])];
            str = 'Class 2 vs Other Classes';
        case 3
            x = [Data(:, 21:30), Data(:, [1:20, 31:end])];
            str = 'Class 3 vs Other Classes';
        case 4
            x = [Data(:, 31:end), Data(:, 1:30)];
            str = 'Class 4 vs Other Classes';
    end
    
    fig1 = figure(1);
    axis square
    xlim([-12, 12]);
    ylim([-12, 12]);    
    plot(x(2, 1:10), x(3, 1:10), 'or', 'MarkerFaceColor', 'r');
    plot(x(2, 11:end), x(3, 11:end), 'ob', 'MarkerFaceColor', 'b');
    title(str);
    
    rnd = @(a, b)(a + (b - a) .* rand());
    a = -0.01;
    b = +0.01;
    K = 2;
    d = 2;
    N = 40;
    c = 2.0;
    eta = c;
    w = zeros(K, d+1);
    delta_w = zeros(K, d+1);
    o = zeros(K, 1);
    y = zeros(K, 1);
    r = zeros(K, N);
    Y = zeros(K, N);
    r(1, 1:10) = 1;
    r(2, 11:end) = 1;
    
    for i = 1:K
        for j = 1:d + 1
            w(i, j) = rnd(a, b);
        end
    end
    
    iter = 1;
    while true
        wOld = w;
        for i = 1:K
            for j = 1:d + 1
                delta_w(i, j) = 0;
            end
        end
        for t = 1:N
            for i = 1:K
                o(i) = 0;
                for j = 1:d + 1
                    o(i) = o(i) + w(i, j) .* x(j, t);
                end
            end
            for i = 1:K
                y(i) = exp(o(i)) / sum(exp(o(:)));
            end
            for i = 1:K
                for j = 1:d + 1
                    delta_w(i, j) = delta_w(i, j) + (r(i, t) - y(i)) .* x(j, t);
                end
            end
            Y(:, t) = y(:);
        end
        for i = 1:K
            for j = 1:d + 1
                w(i, j) = w(i, j) + eta .* delta_w(i, j);
            end
        end
        
        w(1, :) = w(1, :) ./ norm(w(1, 2:3));
        w(2, :) = w(2, :) ./ norm(w(2, 2:3));
        wNew = w;
        
        if all(ismembertol(wOld, wNew, 0.000001, 'ByRows', true))
            break;
        end
        disp(w);
        iter = iter + 1;
        eta = c / iter;
    end
    
    plt0 = ezplot([num2str(w(1, 2)), '* x1 +', num2str(w(1, 3)), '* x2 +', num2str(w(1, 1))], [-20, 20]);
    set(plt0, 'Color', 'magenta', 'LineWidth', 1);
    saveas(fig1, ['plot_2_', num2str(s), '.png']);
    
    fig2 = figure(2);
    width = 0.4;
    b = bar3(Y);
    for k = 1:length(b)
        zdata = b(k).ZData;
        b(k).CData = zdata;
        b(k).FaceColor = 'interp';
    end
    title(str);
    saveas(fig2, ['bar3_2_', num2str(s), '.png']);
    
    pause(0.25);  
    tW = [tW; w(1, :)];
    
    close all;    
end

fig3 = figure(3);
hold on;
axis square
xlim([-12, 12]);
ylim([-12, 12]);
plot(Data(2,  1:10), Data(3, 1:10), 'or', 'MarkerFaceColor', 'r');
plot(Data(2, 11:20), Data(3, 11:20), 'og', 'MarkerFaceColor', 'g');
plot(Data(2, 21:30), Data(3, 21:30), 'ob', 'MarkerFaceColor', 'b');
plot(Data(2, 31:40), Data(3, 31:40), 'oc', 'MarkerFaceColor', 'c');
plt1 = ezplot([num2str(tW(1, 2)), '* x1 +', num2str(tW(1, 3)), '* x2 +', num2str(tW(1, 1))], [-12, 12]);
set(plt1, 'Color', 'magenta', 'LineWidth', 1);
plt2 = ezplot([num2str(tW(2, 2)), '* x1 +', num2str(tW(2, 3)), '* x2 +', num2str(tW(2, 1))], [-12, 12]);
set(plt2, 'Color', 'magenta', 'LineWidth', 1);
plt3 = ezplot([num2str(tW(3, 2)), '* x1 +', num2str(tW(3, 3)), '* x2 +', num2str(tW(3, 1))], [-12, 12]);
set(plt3, 'Color', 'magenta', 'LineWidth', 1);
plt4 = ezplot([num2str(tW(4, 2)), '* x1 +', num2str(tW(4, 3)), '* x2 +', num2str(tW(4, 1))], [-12, 12]);
set(plt4, 'Color', 'magenta', 'LineWidth', 1);
title([]);
saveas(fig3, 'plot_3.png');