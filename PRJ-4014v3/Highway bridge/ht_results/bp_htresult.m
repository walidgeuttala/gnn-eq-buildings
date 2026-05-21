clc
clear all;
close all;

T = readtable('HyperResult.xlsx');
most_common = mode(T{:,3:6})
T = T(T.TL_r2 >= 0.8,:);
most_common = mode(T{:,3:6})


lookbackOrder = [3 5 10 25 50];
x1stOrder = [5,25,50,100,200];
x2ndOrder = [5,25,50,100,200];
fnnOrder = [5,10,20,30,40];

T.lookback = categorical(T.lookback,lookbackOrder);
T.x1st = categorical(T.x1st,x1stOrder);
T.x2nd = categorical(T.x2nd,x2ndOrder);
T.fnn = categorical(T.fnn,fnnOrder);

colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30"];

n_lookback = length(lookbackOrder);
n_x1st = length(x1stOrder);
n_x2nd = length(x2ndOrder);
n_fnn = length(fnnOrder);

%% 1st LSTM layer

f_1st = figure('units','pixels','Position',[0,0,1500,350]);
t_1st = tiledlayout(1,5,'TileSpacing','loose');
xlabel(t_1st,'Units in the first LSTM layer','fontname','Helvetica','fontsize',20)
ylabel(t_1st,'R^2','fontname','Helvetica','fontsize',20)

for i = 1:n_lookback
    nexttile
    data = T(T.lookback==num2str(lookbackOrder(i)),:);

    for j = 1:n_x1st
        data1 = data(data.x1st==num2str(x1stOrder(j)),:);
        b = boxchart(data1.x1st,data1.TL_r2);
        b.BoxFaceAlpha = 0.8;
        b.WhiskerLineColor = 'black';
        b.BoxLineColor = 'black';
        hold on
    end

    ax = gca;
    ax.FontName = 'Helvetica';
    ax.FontSize = 15;

    ax.XLabel.FontSize = 20;
    ax.XGrid = 'off';
    
    ax.YLim = [0.5 1];
    ax.YTick = [0.5:0.1:1];
    ax.YLabel.FontSize = 20;
    ax.YGrid = 'on';

    title(sprintf('Lookback =  %d steps', lookbackOrder(i)),'FontWeight', 'Normal','FontSize',17,'FontName','Helvetica');
end

print(gcf,'1stLSTM','-dpng','-r500');
% close(f_1st)

%% 2nd LSTM layer

f_2nd = figure('units','pixels','Position',[0,0,1500,350]);
t_2nd = tiledlayout(1,5,'TileSpacing','loose');
xlabel(t_2nd,'Units in the second LSTM layer','fontname','Helvetica','fontsize',20)
ylabel(t_2nd,'R^2','fontname','Helvetica','fontsize',20)

for i = 1:n_lookback
    nexttile
    data = T(T.lookback==num2str(lookbackOrder(i)),:);

    for j = 1:n_x2nd
        data1 = data(data.x2nd==num2str(x2ndOrder(j)),:);
        b = boxchart(data1.x2nd,data1.TL_r2);
        b.BoxFaceAlpha = 0.8;
        b.WhiskerLineColor = 'black';
        b.BoxLineColor = 'black';
        hold on
    end

    ax = gca;
    ax.FontName = 'Helvetica';
    ax.FontSize = 15;

    ax.XLabel.FontSize = 20;
    ax.XGrid = 'off';
    
    ax.YLim = [0.5 1];
    ax.YTick = [0.5:0.1:1];
    ax.YLabel.FontSize = 20;
    ax.YGrid = 'on';

    title(sprintf('Lookback =  %d steps', lookbackOrder(i)),'FontWeight', 'Normal','FontSize',17,'FontName','Helvetica');
end

print(gcf,'2ndLSTM','-dpng','-r500');
% close(f_2nd)

%% fnn layer

f_fnn = figure('units','pixels','Position',[0,0,1500,350]);
t_fnn = tiledlayout(1,5,'TileSpacing','loose');
xlabel(t_fnn,'Units in the fully connected layer','fontname','Helvetica','fontsize',20)
ylabel(t_fnn,'R^2','fontname','Helvetica','fontsize',20)

for i = 1:n_lookback
    ax = nexttile;
    data = T(T.lookback==num2str(lookbackOrder(i)),:);

    for j = 1:length(fnnOrder)
        data1 = data(data.fnn==num2str(fnnOrder(j)),:);
        b = boxchart(data1.fnn,data1.TL_r2);
        b.BoxFaceAlpha = 0.8;
        b.WhiskerLineColor = 'black';
        b.BoxLineColor = 'black';
        hold on
    end

%     ax = gca;
    ax.FontName = 'Helvetica';
    ax.FontSize = 15;

    ax.XTickLabel = ["","10","20","30",""];
    ax.XLabel.FontSize = 20;
    ax.XGrid = 'off';
    
    ax.YLim = [0.5 1];
    ax.YTick = [0.5:0.1:1];
%     ax.YLabel.String = {'R^2'};
    ax.YLabel.FontSize = 20;
    ax.YGrid = 'on';

    title(sprintf('Lookback =  %d steps', lookbackOrder(i)),'FontWeight', 'Normal','FontSize',17,'FontName','Helvetica');
end

print(gcf,'FNN','-dpng','-r500');
% close(f_fnn)