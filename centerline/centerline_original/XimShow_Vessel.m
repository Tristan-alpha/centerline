function XimShow_Vessel( imset,Thres )

Ximflag = 1;
nx = size(imset,1);
ny = size(imset,2);
nz = size(imset,3);

% Set graphics parameters.
% fig = figure;
% set(gca,'color','k')

axis([-1 1 -1 1 -1 1]);
caxis(26.9*[-1.5 1]);
colormap(hot);
axis off

pos=get(gca,'outerposition');
left=pos(1);
bottom=pos(2);
width=pos(3);
height=pos(4);

% Buttons
% uicontrol('pos',[width-70 20   60 20],'string','done','fontsize',12,         'callback','close(gcbf);Ximflag=0;');
uicontrol('pos',[width+nx-60 bottom+200   60 20],'string','Previous','fontsize',12,     'callback',{@Prebutton_Callback});
uicontrol('pos',[width+nx-60 bottom+225   60 20],'string','Next','fontsize',12,         'callback',{@Nextbutton_Callback});
uicontrol('pos',[width+nx-60 bottom+250   60 20],'string','Auto','fontsize',12,         'callback',{@Autobutton_Callback});
% Run
i = 1;
dt = 3;
set(gca,'userdata',i);
imshow(imset(:,:,1),'DisplayRange',Thres );
title(['SLICE ',num2str(i)],'color','k','fontsize',16);

autoflag=1;


    function Prebutton_Callback(source,eventdata)
        dt =3 ;
        i = i-dt;
        if(i==0)
            i=nz;
        end
        im = imset(:,:,i);
%         figure(fig);
        imshow(im,'DisplayRange',Thres );
        autoflag=0;
        title(['SLICE ',num2str(i)],'color','k','fontsize',16);
        set(gcbf,'userdata',i);
        drawnow
    end

    function Nextbutton_Callback(source,eventdata)
        i = i+dt;
        if(i>nz)
            i=1;
        end
        im = imset(:,:,i);
%         figure(fig);
        imshow(im,'DisplayRange',Thres );
        autoflag=0;
        title(['SLICE ',num2str(i)],'color','k','fontsize',16);
        set(gcbf,'userdata',i);
        drawnow
        
    end

    function Autobutton_Callback(source,eventdata)
        autoflag = 1;
        while  autoflag
            dt = 1;
            i = i+dt;
            if(i>nz)
                i = 1;autoflag = 0;
            end
            im = imset(:,:,i);
%             figure(fig);
            imshow(im,'DisplayRange',Thres );
            title(['SLICE ',num2str(i)],'color','k','fontsize',16);
            drawnow
        end
        
    end
end

