function plotbm(Image, Image_adv, class_pre, class_adv, fval, fopt)
set(0, 'DefaultFigureRenderer', 'painters');
lb1 = sprintf('Predict Number: %d',class_pre);
lb2 = sprintf('Predict Number: %d',class_adv);
lb3 = sprintf('Distance: %f',norm(Image_adv-Image,'fro'));
imsize = sqrt(numel(Image));
figure;
colormap gray
set(gcf,'position',[200,200,800,640])

subplot('Position',[0.025 0.63 0.3 0.33])
imagesc(reshape(Image,imsize,imsize));
set(gca,'xtick',[],'ytick',[])
title('Original Image','FontSize',20)
xlabel(lb1,'FontSize',18)

subplot('Position',[0.35 0.63 0.3 0.33])
imagesc(reshape(Image_adv,imsize,imsize));
set(gca,'xtick',[],'ytick',[])
title('Adversarial Attack','FontSize',20)
xlabel(lb2,'FontSize',18)

subplot('Position',[0.675 0.63 0.3 0.33])
imagesc(reshape(Image_adv-Image,imsize,imsize));
set(gca,'xtick',[],'ytick',[])
title('Difference','FontSize',20)
xlabel(lb3,'FontSize',18)

Illini_Orange = '#DD3403';
Illini_Blue   = '#13294B';
Stanford_Red  = '#8C1515';

% whether to create a subplot for condition number
pos = [0.1 0.09 0.8 0.45];
% for lenged posistion
minval = inf; maxval = -inf;
for k = 1:numel(fval)
    minval = min([fval{k}.bm,minval]); maxval = max([fval{k}.bm,maxval]); 
end
legpos = 'sw';

subplot('Position',pos)
hold on
grid on
base = 0;
for k = 1:numel(fval)
    h1 = plot(base+(0:fval{k}.iter),[fval{k}.bm(1),fval{k}.bm],'Color',Illini_Blue,'LineStyle','-','LineWidth',2.5);
    plot([base,base],[minval-0.2*abs(maxval-minval),maxval+0.1*abs(maxval-minval)],'Color',Stanford_Red,'LineStyle','--','LineWidth',2);
    text(base,maxval+0.1*abs(maxval-minval),['$$\ r=',num2str(fval{k}.r),'$$'],'Color',Stanford_Red,'interpreter','latex','FontSize',16,...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
    base = base + fval{k}.iter;
end
legend(h1,'BM','location',legpos,'FontSize',22);

set(gca,'fontsize',16)
title('Objective Value','FontSize',20)
xlabel('Iterations','FontSize',18)
ylim([minval-0.2*abs(maxval-minval),maxval+0.2*abs(maxval-minval)])
end