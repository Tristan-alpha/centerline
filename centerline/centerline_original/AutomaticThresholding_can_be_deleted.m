function [MaxVal,MinVal]=AutomaticThresholding_can_be_deleted(image_all,Pos,showflag)
%
Len=20;
for cont=1:Len
    Val{cont}=image_all(Pos(1)-cont:Pos(1)+cont,Pos(2)-cont:Pos(2)+cont,Pos(3)-cont:Pos(3)+cont);
    %     Min(cont)=min(Val{cont}(:));
    %     Max(cont)=max(Val{cont}(:));
    %     Mean(cont)=mean(Val{cont}(:));
    %     Std(cont)=std(Val{cont}(:));
end
GrayMin = 900;
GrayMax = 2000;
% Interval=[GrayMin,GrayMax];
GrayPixelSelected=image_all(Pos(1),Pos(2),Pos(3));
IndPixelSelected = GrayPixelSelected-GrayMin+1;


Isfilter = 0;
Map=[];

HistoInterval=5;
for cont=1:Len
    % Val(Val<Interval(1))=Interval(1);
    % Val(Val>Interval(2))=Interval(2);
    Vec=hist(Val{cont}(:),900:HistoInterval:2000)/length(Val{cont}(:));
    
    
    w = gausswin(10);
    w = w/sum(w);
    Vec1 = filter(w,1,Vec);
    % Vec=smooth(Vec,20);
    
    Map1(:,cont)=Vec1;
    Map(:,cont)=Vec;
end




if showflag
    figure,
    subplot(5,1,1)
    % plot((GrayMin:HistoInterval:GrayMax)-1024,Map(:,1));
    bar((GrayMin:HistoInterval:GrayMax)-1024,Map(:,1));
    set(gca,'ytick',[])
    set(gca,'xlim',[GrayMin GrayMax]-1024)
    % xlabel('pixel intensity (unit, HU)','fontsize',20)
    ylabel('r=1','fontsize',20)
    
    subplot(5,1,2)
    % plot((GrayMin:HistoInterval:GrayMax)-1024,Map(:,5));
    bar((GrayMin:HistoInterval:GrayMax)-1024,Map(:,5));
    set(gca,'ytick',[])
    set(gca,'xlim',[GrayMin GrayMax]-1024)
    % xlabel('pixel intensity (unit, HU)')
    ylabel('r=5','fontsize',20)
    
    subplot(5,1,3)
    % plot((GrayMin:HistoInterval:GrayMax)-1024,Map(:,10));
    bar((GrayMin:HistoInterval:GrayMax)-1024,Map(:,10));
    set(gca,'ytick',[])
    set(gca,'xlim',[GrayMin GrayMax]-1024)
    % xlabel('pixel intensity (unit, HU)')
    ylabel({'histogram', 'r=10'},'fontsize',20)
    
    subplot(5,1,4)
    % plot((GrayMin:HistoInterval:GrayMax)-1024,Map(:,15));
    bar((GrayMin:HistoInterval:GrayMax)-1024,Map(:,15));
    set(gca,'ytick',[])
    set(gca,'xlim',[GrayMin GrayMax]-1024)
    % xlabel('pixel intensity (unit, HU)','fontsize',20)
    ylabel('r=15','fontsize',20)
    
    subplot(5,1,5)
    % plot((GrayMin:HistoInterval:GrayMax)-1024,Map(:,20));
    bar((GrayMin:HistoInterval:GrayMax)-1024,Map(:,20));
    set(gca,'ytick',[])
    set(gca,'xlim',[GrayMin GrayMax]-1024)
    xlabel('pixel intensity (unit, HU)','fontsize',20)
    ylabel('r=20','fontsize',20)
    
    
    
    
    figure('Position',[100,500,600,600]),
    subplot(2,1,1)
    imagesc(Map'),
    xlabel('histogram (i.e., pdf)','fontsize',20)
    ylabel('ROI size (radius)','fontsize',20)
    set(gca,'xtick',[])
    % set(gca,'ytick',[1 5 10 15 20])
    set(gca,'ytick',[])
    % set(gca,'xlim',[GrayMin GrayMax]-1024)
    
    
    
    
    
%     figure,
%     subplot(2,1,1)
%     bar((GrayMin:HistoInterval:GrayMax)-1024,sum(Map')/20);
%     % xlabel('pixel intensity (unit, HU)','fontsize',20)
%     ylabel('weighted histogram','fontsize',20)
%     %     imagesc(Map'),
%     %     xlabel('histogram (i.e., pdf)')
%     %     ylabel('ROI size')
%     %     set(gca,'xtick',[])
%     %
%     set(gca,'ytick',[])
%     %     % colormap gray
%     %     grid on
    
    
    
    subplot(2,1,2)
    plot((GrayMin:HistoInterval:GrayMax)-1024,sum(Map1')/20)
    % set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca,'xlim',[GrayMin GrayMax]-1024)
    xlabel('pixel intensity (unit, HU)','fontsize',20)
    ylabel('weighted histogram curve','fontsize',20)
end
%%
Histo=sum(Map1');
[pks,locs,w,p] = findpeaks(Histo,GrayMin:HistoInterval:GrayMax);
[Val,Ind]=min(abs(locs-GrayPixelSelected));
Center=locs(Ind);
Width=round(w(Ind));
if showflag
    % findpeaks(Histo,'MinPeakProminence',0.4,'Annotate','extents')
    figure('Position',[100,500,600,300]),findpeaks(Histo,(GrayMin:HistoInterval:GrayMax)-1024,'MinPeakProminence',0.15,'Annotate','extents')
    legend off
    % set(gca,'xtick',[])
    set(gca,'ytick',[])
    hold on
    
    
    plot(Center,pks(Ind),'ro');
    Y=ylim();
    plot([Center-2*Width,Center-2*Width]-1024,Y,'m');
    plot([Center-4*Width,Center-4*Width]-1024,Y,'m');
    plot([Center+2*Width,Center+2*Width]-1024,Y,'m');
    plot([Center+4*Width,Center+4*Width]-1024,Y,'m');
    xlabel('pixel intensity (unit, HU)','fontsize',20)
end

MaxVal=Center+2*Width;
MinVal=Center-4*Width;





% CenterCurve = Map(IndPixelSelected,:);
%
% for cont=1:size(Map,1)
%     Dif(cont,:) = abs(Map(cont,:)-CenterCurve);
%     Loss(cont)=norm(Map(cont,:)-CenterCurve);
% end
% figure,
% imagesc(Dif)
% grid on
%
% figure,plot(Loss)