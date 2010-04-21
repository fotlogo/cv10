function [] = test()

%---------------------------------------
% get attributes and bounding boxes for
% data
%---------------------------------------
fname = 'data/attribute_data/ayahoo_test.txt';
imdir = 'data/ayahoo_test_images';
hogdir = 'out/ayahoo_test_images/processed/hog';
[img_names img_classes bboxes attributes] = read_att_data(fname);

%---------------------------------------
% get the attribute names
%---------------------------------------
fid = fopen('data/attribute_data/attribute_names.txt');
[atts] = textscan(fid, '%s', 'delimiter', '\n');
atts = atts{1};
fclose(fid);

%---------------------------------------
% get the class names
%---------------------------------------
fid = fopen('data/attribute_data/class_names.txt');
[classes] = textscan(fid, '%s', 'delimiter', '\n');
classes = classes{1};
fclose(fid);

%---------------------------------------
% Load all of the features for the
% dataset.
%---------------------------------------
count = 100;%length(img_names);
rand('seed',1);
testindices = (rand(1, count) < 0.2);
%testindices = (rand(1, count) < 0.02);
numfeatures = 1000;
numatts = size(atts, 1);

imgfeat = zeros(count, numfeatures);
imglabel = zeros(count, 1);
imgatt = zeros(count, numatts);

j = 1;
TRAIN = 0;
for i = 1:count
  if (testindices(i) | TRAIN)
    im = imread(fullfile(imdir, img_names{i}));

    % [x_min y_min x_max y_max]
    box = bboxes(i,:);
    
    img_name = regexprep(char(img_names(i)), '\.jpg', '');
    hogfn = sprintf('%s/%s_hog.mat', hogdir, img_name);
    hog = load(hogfn);
    hog_ind = hog.x >= box(1) & hog.y >= box(2) & hog.x <= box(3) & hog.y <= box(4);

    feat_hog = hist(hog.idx(hog_ind),1:numfeatures);
    feat_hog_norm = feat_hog/norm(feat_hog);
    imgfeat(i,:) = feat_hog;
    imglabel(i,:) = find(strcmp(classes, img_classes{i}));
    imgatt(i,:) = attributes(i,:);

    %feat_color = hist(tc.colorim(box(2):box(4), box(1):box(3)),1:128);
    %feat_color_norm = feat_color/norm(feat_color);

    %feat_texture = hist(tc.textonim(box(2):box(4), box(1):box(3)),1:256);
    %feat_texture_norm = feat_texture/norm(feat_texture);

    disp(sprintf('%d %s %s', i, img_classes{i}, img_name));

    j = j + 1;
  end
end

addpath('libsvm')

%---------------------------------------
% Split the feature vectors into train
% and test data
%---------------------------------------
imgfeat_train = imgfeat(find(~testindices),:);
imgfeat_test = imgfeat(find(testindices),:);
imglabel_train = imglabel(find(~testindices),:);
imglabel_test = imglabel(find(testindices),:);
imgatt_train = imgatt(find(~testindices),:);
imgatt_test = imgatt(find(testindices),:);

%---------------------------------------
% train a classifier for each attribute
%---------------------------------------
if (TRAIN)
  for i = 1:numatts
    att = imgatt_train(:,i);
    %imglabel_train = imgatt_train(:,i);
    %imglabel_test = imgatt_test(:,i);

    model = svmtrain(att, imgfeat_train, '') ;
    if (i == 1)
      models = model;
    else
      models(i) = model;
    end

    %att = imgatt_train(:,i);
    att = imgatt_test(:,i);
    %feat = imgfeat_train(:,i);
    feat = imgfeat_test(:,i);
    [predict_label, accuracy, dec_values] = svmpredict(att, feat, model) ;
  end
  %save('models.mat', 'models');
  save('models_small.mat', 'models');
else
  disp(sprintf('loading models.mat...'));
  %load('models.mat');
  load('models_small.mat');
end

%---------------------------------------
% See what attributes we get for the 
% first image
%---------------------------------------
imgidx = find(testindices, 1);
atthist = zeros(1, numatts);
for i = 1:numatts
  imgfeat_test = imgfeat(imgidx,i);
  imgatt_test = imgatt(imgidx,i);
  [predict_label, accuracy, dec_values] = svmpredict(imgatt_test, imgfeat_test, models(i));
  atthist(i) = predict_label;
end
disp(sprintf('%u', atthist));
disp(sprintf('%u', imgatt(imgidx,:)));
