function [Center,St,End]=FindThePeak(Signal,Peakpos,showflag)
[pks,locs,w,p] = findpeaks(Signal);
[pks_,locs_,w_,p_] = findpeaks(-Signal);
%
if showflag
    figure,plot(Signal),
    hold on,plot(locs,pks,'r*')
    plot(locs_,-pks_,'g*')
end
%
[Val,Ind]=min(abs(locs-Peakpos));
Center=locs(Ind);
St=locs_(max(find(locs_<Peakpos)));
End=locs_(min(find(locs_>Peakpos)));
%
if showflag
    plot(Center,Signal(Center),'ro');
    plot(St,Signal(St),'go');
    plot(End,Signal(End),'go');
end



















