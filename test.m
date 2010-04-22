%------------------------------------------------------------------
% Testing function for Aibo and John's vision project.  This
% function 
%   * reads test and training images
%   * loads the image names, classes, bounding boxes and attributes
%   * extracts features (probably already pre-processed)
%   * optionally trains svm classifiers for the attributes
%   * optionally tests the classifiers
%
% Notes:
%   * The variable 'count' controls how many images are used
%     - 1/10 of 'count' are used for testing
%   * The variable 'numatts' controls how many attributes are
%     learned and tested
%   * The variable 'numfeats' controls how many features are
%     used.  When using only hog, 1000 should be used.  This
%     should be increased as we add more fetaures.
%   * Variables 'TRAIN' and 'TEST' control whether training and
%     testing are executed
%   * multiclass libsvm is currently used - but we don't need
%     multiclass!
%------------------------------------------------------------------
function [] = test()
addpath('grouping');
addpath('grouping/lib');
addpath('libsvm')

gPbdir = 'out/ayahoo_test_images/processed/gPb';

%---------------------------------------
% get attributes and bounding boxes for
% data
%---------------------------------------
fname = 'data/attribute_data/ayahoo_test.txt';
imdir = 'data/ayahoo_test_images';
hogdir = 'out/ayahoo_test_images/processed/hog';
tcdir = 'out/ayahoo_test_images/processed/tc2';
[img_names img_classes bboxes attributes] = read_att_data(fname);

% Takes the first 'count' images
count = 100;%length(img_names);
rand('seed',1);
test_indices = (rand(1, count) < 0.2);
%test_indices = (rand(1, count) < 0.02);
train_indices = ~test_indices;

% Takes a random set of roughly 'count' images
perc = count / length(img_names);
train_indices = (rand(1, length(img_names)) < perc);
test_indices = (rand(1, length(img_names)) < perc / 10);

%---------------------------------------
% split image names and classes up into
% training and testing sets
%---------------------------------------
names_train = img_names(find(train_indices),:);
names_test = img_names(find(test_indices),:);
classes_train = img_classes(find(train_indices),:);
classes_test = img_classes(find(test_indices),:);
bboxes_train = bboxes(find(train_indices),:);
bboxes_test = bboxes(find(test_indices),:);
attributes_train = attributes(find(train_indices),:);
attributes_test = attributes(find(test_indices),:);

count_train = size(names_train, 1);
count_test = size(names_test, 1);

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
% dataset
%---------------------------------------
%numfeatures = 1384;
numfeatures = 1000;
numatts = size(atts, 1);

TRAIN = 0;
TEST = 1;

features_train = zeros(count_train, numfeatures);
labels_train = zeros(count_train, 1);

%---------------------------------------
% train a classifier for each attribute
%---------------------------------------
if (TRAIN)
  % get the features for training images
  for i = 1:count_train
    img_name = regexprep(char(names_train(i)), '\.jpg', '');
    [feat]  = get_features(imdir, img_name, ...
				  hogdir, tcdir, ...
				  bboxes_train(i,:));
    features_train(i,:) = feat;
    labels_train(i,:) = find(strcmp(classes, classes_train{i}));
    disp(sprintf('%d %s %s', i, classes_train{i}, img_name));
  end

  % train the classifier
  att_pred = zeros(count_train, numatts);
  for i = 1:numatts
    att = attributes_train(:,i);

    model = svmtrain(att, features_train, '');
    if (i == 1)
      models = model;
    else
      models(i) = model;
    end

    att = attributes_train(:,i);
    feat = features_train(:,i);
    [predict_label, accuracy, dec_values] = svmpredict(att, feat, model);
    %[T, predict_label, accuracy, dec_values] = evalc('svmpredict(att, feat, model)');
    att_pred(:, i) = predict_label;
  end
  %disp('Predicted attributes');
  %for i = 1:size(att_pred, 1)
  %  disp(sprintf('%u', att_pred(i,:)));
  %end
  %save('models.mat', 'models');
  save('models_small.mat', 'models');

else
  % if we're not training, load the classifiers from disk
  disp(sprintf('loading models.mat...'));
  %load('models.mat');
  load('models_small.mat');
end

%---------------------------------------
% test the classifiers
%---------------------------------------
if (TEST)
  %---------------------------------------
  % get features for test images
  %---------------------------------------
  features_test = zeros(count_test, numfeatures);
  labels_test = zeros(count_test, 1);

  % get the features for test images
  for i = 1:count_test
    img_name = regexprep(char(names_test(i)), '\.jpg', '');
    [feat]  = get_features(imdir, img_name, ...
				  hogdir, tcdir, ...
				  bboxes_test(i,:));
    features_test(i,:) = feat;
    labels_test(i,:) = find(strcmp(classes, classes_test{i}));
    disp(sprintf('%d %s %s', i, classes_test{i}, img_name));
  end

  %---------------------------------------
  % See what attributes we get for the 
  % first image
  %---------------------------------------
  for imgidx = 1:50:count_test %find(test_indices, 1);
  att_pred = zeros(1, numatts);
  att_actual = attributes_test(imgidx,:);
  features = features_test(imgidx,:);
  for i = 1:numatts
    [T, predict_label, accuracy, dec_values] = evalc('svmpredict(att_actual(i), features, models(i))');
    att_pred(i) = predict_label;
    %disp(sprintf('%u', att_pred));
  end
  disp(sprintf('%u', att_pred));
  disp(sprintf('%u', att_actual));
  %disp(sprintf('%f', features));
  end
end