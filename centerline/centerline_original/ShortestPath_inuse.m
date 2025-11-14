function [RouteMat,pathOut]=ShortestPath_inuse(image_input,TopPoint,MidPoint,J,outputPath,LeftOrRight,showflag)
K=4;
D3data = exp(K*double(image_input));
MaxVal=max(D3data(:));
% save image_all.mat image_all
% clear image_all

% ConvKer=ones(4,4,4)/(4^3);
% D3data=convn(D3data,ConvKer,'same').*(image_input>0);
% for cont=1:size(TopPoint,1)
%     D3data(TopPoint(cont,1),TopPoint(cont,2),1)=double(image_input(TopPoint(cont,1),TopPoint(cont,2),1));
% end
% D3data(TopPoint(cont,1),TopPoint(cont,2),1)=double(image_input(MidPoint(cont,1),MidPoint(cont,2),end));


% 造出映射表
Mark=1;
LineIndex=zeros(size(D3data));
for cont=1:size(D3data,1)
    for cont1=1:size(D3data,2)
        for cont2=1:size(D3data,3)
            if D3data(cont,cont1,cont2)~=MaxVal
                % 不为零的才能入表
                Pos{Mark}=[cont,cont1,cont2];%从指标到坐标的映射
                LineIndex(cont,cont1,cont2)=Mark;%从坐标到指标的映射
                Mark=Mark+1;
            end
        end
    end
end
% 计算权值图
% Beta=3;
% D3data=(D3data-min(D3data(:)))/(max(D3data(:))-min(D3data(:)));

Neighbour=1;
SparLen=(((Neighbour*2+1)^3-1)/2+1)*length(Pos);
Xindex=zeros(1,SparLen);
Yindex=zeros(1,SparLen);
Zvalue=zeros(1,SparLen);
Mark=1;


for cont2=1:size(D3data,3)
    for cont1=1:size(D3data,2)
        for cont=1:size(D3data,1)
            if D3data(cont,cont1,cont2)~=MaxVal
                Neib=findNeib(size(D3data),[cont,cont1,cont2],Neighbour,sqrt(3));
                % 不为零的才能入表
                Index0=LineIndex(cont,cont1,cont2);
                Value1=D3data(cont,cont1,cont2);
                for cont3=1:size(Neib,1)
                    Index1=LineIndex(Neib(cont3,1),Neib(cont3,2),Neib(cont3,3));
                    Value2=D3data(Neib(cont3,1),Neib(cont3,2),Neib(cont3,3));
                    if (Value2~=MaxVal)&&(Index0~=Index1)
                        %邻域的点的取值也不为零
                        Zvalue(Mark)=(Value1+Value2)/2;
                        Xindex(Mark)=Index0;
                        Yindex(Mark)=Index1;
                        Mark=Mark+1;
                    end
                end
            end
        end
    end
    cont2,
end

Xindex(Mark:end)=[];
Yindex(Mark:end)=[];
Zvalue(Mark:end)=[];
% Zvalue=exp(Beta.*Zvalue)-1;

% save Temp.mat D3data LineIndex Pos
% clear D3data LineIndex Pos
G=sparse(Xindex,Yindex,Zvalue);
clear Xindex Yindex Zvalue
% load Temp.mat
% delete('Temp.mat')


% 所有需要标出血管的列表
AllRoute=zeros(length(Pos),1);
path=[];
% 提取四根血管路径出来
Node=LineIndex(TopPoint(1),TopPoint(2),1);
Nv1=LineIndex(MidPoint(1),MidPoint(2),end);


% modification
% [dist1,path1,pred]=graphshortestpath(G,Node,Nv1,'Directed',false);
Gg = graph(G); % 如果是有向图用 digraph(G)
[path1, dist1] = shortestpath(Gg, Node, Nv1);

path=path1;
AllRoute(path)=1;

pathOut=[];
for cont=1:length(path)
    pathOut=[pathOut;Pos{path(cont)}];
end


%%
RouteMat=image_input;
RouteMat(:)=0;
for cont=1:size(pathOut,1)
    RouteMat(pathOut(cont,1),pathOut(cont,2),pathOut(cont,3))=cont;
end
% 显示血管 上面的代码与下面的代码功能完全相同
% RouteMat=image_input;
% RouteMat(:)=0;
% for cont=1:length(AllRoute)
%     if AllRoute(cont)==1
%         RouteMat(Pos{cont}(1),Pos{cont}(2),Pos{cont}(3))=1;
%     end
% end
%% 
if showflag
    for cont=1:size(image_input,3)
        figure(1)
        imshow(J(:,:,cont),[]);
        hold on
        [XX,YY]=find(RouteMat(:,:,cont)~=0);
        plot(YY,XX,'r.');
        hold off
        if LeftOrRight==0
            saveas(gcf,[outputPath,'\Left\Mask','.jpg'],'jpg')
        elseif LeftOrRight==1
            saveas(gcf,[outputPath,'\Right\Mask','.jpg'],'jpg')
        elseif LeftOrRight==-1
            % 另一种备份给packaging函数用的情形
            saveas(gcf,[outputPath,'\Mask',num2str(cont),'.jpg'],'jpg')
        end
        clf
    end
end


