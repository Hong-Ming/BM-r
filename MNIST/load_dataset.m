function [images_test,labels_test] = load_dataset(W,b)

dataloader = load('dataset.mat');
images_test = dataloader.test_images/255;
labels_test = dataloader.test_labels;
pre = predict(W,b,images_test)==labels_test;
images_test = images_test(:,pre);
labels_test = labels_test(pre);


end