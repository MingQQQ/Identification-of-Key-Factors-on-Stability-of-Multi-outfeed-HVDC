%h=imhist(X1,20);
bar(X1)
%hLegend = legend('string',{'S_{1}', 'S_{T}','S_{X}'});
%hTitle = title('Histogram Plot'); 
%hXLabel = xlabel('XAxis'); 

%hYLabel = ylabel('Sensitivity value');%Ӣ��
hYLabel = ylabel('������ֵ','Fontname','����');%����
set(gca,'XtickLabel',{'cbG','cbL','dsG','dsL','lsG','lsL','qnG','qnL','xcG','xcL'})
set(gca,'XtickLabelrotation',45)
%set(gca, 'FontName', 'Times New Roman', 'FontSize', 26) %Ӣ��
set(gca, 'FontName', '����', 'FontSize', 20) %����
%set([hYLabel,hLegend], 'FontName', 'Times New Roman', 'FontSize', 26) %Ӣ��
%set([hYLabel,hLegend], 'FontName', '����', 'FontSize', 26)%����
hLegend = legend({'һ��������', '�ܽ�������','�ֲ�������'}, 'FontSize', 20);
%set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
set(gcf,'unit','centimeters','position',[1 1 18.1 9])
