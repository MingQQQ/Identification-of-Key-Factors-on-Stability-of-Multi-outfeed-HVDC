clc
%maximum power angle difference
LineWidth_Plot = plot(X, B(:,1), 'r',X, B(:,2), 'k',X, B(:,3), 'g',X, B(:,4), 'b',X, B(:,5), 'm');
ylabel('Maximum power angle difference(бу)','Fontname','Times New Roman','FontSize',12);
%highest node voltage
% LineWidth_Plot = plot(X, C(:,1), 'r',X, C(:,2), 'k',X, C(:,3), 'g',X, C(:,4), 'b',X, C(:,5), 'm');
% ylabel('Highest node voltage(p.u.)','Fontname','Times New Roman','FontSize',12);
%highest node frequency
% LineWidth_Plot = plot(X, D(:,1), 'r',X, D(:,2), 'k',X, D(:,3), 'g',X, D(:,4), 'b',X, D(:,5), 'm');
% ylabel('Highest node frequency(Hz)','Fontname','Times New Roman','FontSize',12);
%
legend_FontSize = legend('A','B','C','D','E');
xlabel('T(s)','Fontname','Times New Roman','FontSize',12);
set(gca,'Fontname','Times New Roman','FontSize',12);
set(LineWidth_Plot,'LineWidth', 1);
set(legend_FontSize,'Fontname','Times New Roman','FontSize',12);
set(gcf,'unit','centimeters','position',[1 1 18.1 9])
