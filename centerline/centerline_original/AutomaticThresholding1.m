function [MaxVal,MinVal]=AutomaticThresholding1(image_all,Pos,showflag)
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


Isfilter = 1;
Map=[];

HistoInterval=5;
for cont=1:Len
    % Val(Val<Interval(1))=Interval(1);
    % Val(Val>Interval(2))=Interval(2);
    Vec=hist(Val{cont}(:),900:HistoInterval:2000)/length(Val{cont}(:));
    
    if Isfilter
        w = gausswin(10);
        w = w/sum(w);
        Vec = filter(w,1,Vec);
        % Vec=smooth(Vec,20);
    end
    Map(:,cont)=Vec;
end

if showflag
    figure,
    subplot(2,1,1)
    imagesc(Map'),
    xlabel('histogram (i.e., pdf)')
    ylabel('ROI size')
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    % colormap gray
    grid on
    subplot(2,1,2)
    plot(GrayMin:HistoInterval:GrayMax,sum(Map'))
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca,'xlim',[GrayMin GrayMax])
    xlabel('pixel intensity')
    ylabel('weighted histogram')
end
%%
Histo=sum(Map');
[pks,locs,w,p] = findpeaks(Histo,GrayMin:HistoInterval:GrayMax);
[Val,Ind]=min(abs(locs-GrayPixelSelected));
Center=locs(Ind);
Width=round(w(Ind));
if showflag
    % findpeaks(Histo,'MinPeakProminence',0.4,'Annotate','extents')
    figure,findpeaks(Histo,GrayMin:HistoInterval:GrayMax,'Annotate','extents')
    legend off
    % set(gca,'xtick',[])
    % set(gca,'ytick',[])
    hold on
    
    
    plot(Center,pks(Ind),'ro');
    Y=ylim();
    plot([Center-4*Width,Center-4*Width],Y,'g');
    plot([Center+2*Width,Center+2*Width],Y,'g');
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