%h=imhist(X1,20);
bar(X1)
%hLegend = legend('string',{'S_{1}', 'S_{T}','S_{X}'});
%hTitle = title('Histogram Plot'); 
%hXLabel = xlabel('XAxis'); 

%hYLabel = ylabel('Sensitivity value');%英文
hYLabel = ylabel('灵敏度值','Fontname','宋体');%中文
set(gca,'XtickLabel',{'cbG','cbL','dsG','dsL','lsG','lsL','qnG','qnL','xcG','xcL'})
set(gca,'XtickLabelrotation',45)
%set(gca, 'FontName', 'Times New Roman', 'FontSize', 26) %英文
set(gca, 'FontName', '宋体', 'FontSize', 20) %中文
%set([hYLabel,hLegend], 'FontName', 'Times New Roman', 'FontSize', 26) %英文
%set([hYLabel,hLegend], 'FontName', '宋体', 'FontSize', 26)%中文
hLegend = legend({'一阶灵敏度', '总阶灵敏度','局部灵敏度'}, 'FontSize', 20);
%set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
set(gcf,'unit','centimeters','position',[1 1 18.1 9])
