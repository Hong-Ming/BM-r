function plotpgd(Image, Image_adv, class_true, class_adv, fval, fopt)
set(0, 'DefaultFigureRenderer', 'painters');
lb1 = sprintf('Predict Number: %d',class_true);
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
Starford_Red  = '#8C1515';

subplot('Position',[0.1 0.09 0.8 0.45])
hold on
grid on
plot(1:fval.iter,fval.pgd,'Color',Illini_Orange,'LineStyle','-','LineWidth',2.5);
plot(find(fval.pgd==fopt),fopt,'x','Color',Starford_Red,'MarkerSize',15,'LineWidth',4);
set(gca,'fontsize',16)
title('Objective Value','FontSize',20)
xlabel('Iterations','FontSize',18)
legend('PGD','Optimal','location','ne','FontSize',22);
end