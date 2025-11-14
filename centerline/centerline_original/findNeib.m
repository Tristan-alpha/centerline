function Neib=findNeib(MatSize,Pos,Neighbour,dis)
Neib=[];
for cont=Pos(1)-Neighbour:Pos(1)+Neighbour
    for cont1=Pos(2)-Neighbour:Pos(2)+Neighbour
        for cont2=Pos(3)-Neighbour:Pos(3)+Neighbour
            if (cont>=1)&&(cont<=MatSize(1))&&...
                    (cont1>=1)&&(cont1<=MatSize(2))&&...
                    (cont2>=1)&&(cont2<=MatSize(3))
                if ((cont-Pos(1))^2+(cont1-Pos(2))^2+(cont2-Pos(3))^2)<=(dis^2)
                    Neib=[Neib;cont,cont1,cont2];
                end
            end
        end
    end
end



