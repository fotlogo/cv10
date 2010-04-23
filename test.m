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
%------------------------------------------------------------------

% seeded random number generator before calling randpermute()
% stored svm model so we can run without training
% changed some print output

%function [] = test()
addpath('grouping');
addpath('grouping/lib');
addpath('libsvm')
addpath('SVM-KM')

gPbdir = 'out/ayahoo_test_images/processed/gPb';

%---------------------------------------
% get attributes and bounding boxes for
% data
%---------------------------------------
fname = 'data/attribute_data/ayahoo_test.txt';
global img_dir hog_dir tc_dir;
img_dir = 'data/ayahoo_test_images';
hog_dir = 'out/ayahoo_test_images/processed/hog';
tc_dir = 'out/ayahoo_test_images/processed/tc2';
[img_names img_classes bboxes attributes] = read_att_data(fname);

% change to random of permutation here, Aibo
count_train = 1000;
count_test = 100;
rand('seed', 1);
rand_indices=randperm(length(img_names));
train_indices=rand_indices(1:count_train);
test_indices=rand_indices(count_train+1:count_train+count_test);

%---------------------------------------
% split image names and classes up into
% training and testing sets
%---------------------------------------

names_train = img_names(train_indices,:);
names_test = img_names(test_indices,:);
classes_train = img_classes(train_indices,:);
classes_test = img_classes(test_indices,:);
bboxes_train = bboxes(train_indices,:);
bboxes_test = bboxes(test_indices,:);
attributes_train = attributes(train_indices,:);
attributes_test = attributes(test_indices,:);

% count_train = size(names_train, 1);
% count_test = size(names_test, 1);

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
TEST = 0;

features_train = zeros(count_train, numfeatures);
%labels_train = zeros(count_train, 1);

kernel='gaussian';
kerneloption=5;

%---------------------------------------
% train a classifier for each attribute
%---------------------------------------
if (TRAIN)
    disp ('Start training................')
  % get the features for training images
  for i = 1:count_train
    img_name = regexprep(char(names_train(i)), '\.jpg', '');
    [feat]  = get_features(img_name, bboxes_train(i,:));
    features_train(i,:) = feat;
    %labels_train(i,:) = find(strcmp(classes, classes_train{i}));
    %disp(sprintf('%d %s %s', i, classes_train{i}, img_name));
  end

  % train the classifier
  %att_pred = zeros(count_train, numatts);
  
  % svm km
  % Kernel Parameters
  % -------------------------------------------------------
  C=100000000;
  verbose=0;
  lambda=1e-7;
  nbclass=2;
  % -------------------------------------------------------
  % Solving
  % -------------------------------------------------------
  ypred=[];
  supVec={};
  wVec={};
  bVec={};
  
  for i = 1:numatts
    att = attributes_train(:,i);
    ratio=0.3;  % ratio of positive samples
    att_pos=find(att==1);
    att_neg=find(att==0);
    if ((length(att_neg)/length(att_pos))>((1-ratio)/ratio))
        att_neg=att_neg(1:floor(length(att_pos)/ratio*(1-ratio)));
    end
    if length(att_pos)==0
        features_temp=features_train;
        att_temp=att;
    else
        features_temp=[features_train(att_pos,:);features_train(att_neg,:)];
        att_temp=[att(att_pos,:);att(att_neg,:)];
    end
    
    % svm km
    att_temp(att_temp==0)=-1;
    [xsup,w,b,pos,timeps,alpha,obj]=svmclass(features_temp,att_temp,C,lambda,kernel,kerneloption,verbose);
    supVec=[supVec,xsup];
    wVec=[wVec,w];
    bVec=[bVec,b];
    y=svmval(features_temp,xsup,w,b,kernel,kerneloption);
    y(y>0)=1;
    y(y<=0)=-1;
    disp(sum(y==att_temp)/length(y))
    
%     if (i == 1)
%       models = model;
%     else
%       models(i) = model;
%     end

%     att = attributes_train(:,i);
%     feat = features_train;%(:,i);
%     [predict_label, accuracy, dec_values] = svmpredict(att, feat, model);
%     disp (sum(predict_label))
    %[T, predict_label, accuracy, dec_values] = evalc('svmpredict(att, feat, model)');
    %att_pred(:, i) = predict_label;
  end
  save('models.mat', 'supVec', 'wVec', 'bVec');
%disp('Predicted attributes');
  %for i = 1:size(att_pred, 1)
  %  disp(sprintf('%u', att_pred(i,:)));
  %end
  %save('models.mat', 'models');
  %save('models_small.mat', 'models');

else
  % if we're not training, load the classifiers from disk
  disp(sprintf('loading models.mat...'));
  load('models.mat');
  %load('models_small.mat');
end

%---------------------------------------
% test the classifiers
%---------------------------------------
if (TEST)
  disp ('Start testing ..................')

  %---------------------------------------
  % get features for test images
  %---------------------------------------
  features_test = zeros(count_test, numfeatures);
  %labels_test = zeros(count_test, 1);

  % get the features for test images
  for i = 1:count_test
    img_name = regexprep(char(names_test(i)), '\.jpg', '');
    [feat]  = get_features(img_name, bboxes_test(i,:));
    features_test(i,:) = feat;
    %labels_test(i,:) = find(strcmp(classes, classes_test{i}));
    %disp(sprintf('%d %s %s', i, classes_test{i}, img_name));
  end

  %---------------------------------------
  % See what attributes we get for the 
  % test images
  %---------------------------------------
  att_pred = zeros(count_test, numatts);
  att_actual = attributes_test;
  features = features_test;
  precision=[];
  for i = 1:numatts
    % svm km
    y=svmval(features,supVec{i},wVec{i},bVec{i},kernel,kerneloption);
    y(y>0)=1;
    y(y<=0)=0;
    att_pred(:,i)=y;
    disp(sprintf('%2d: positive = %3d; precision = %1.2f', i, sum(y==1), ...
		 sum(y==att_actual(:,i))/length(y)));
    precision=[precision,sum(y==att_actual(:,i))/length(y)];
  end
  disp(sprintf('total precision: %1.2f', sum(precision)/length(precision)));
end

SEGMENTATION = 0;
if (SEGMENTATION)
temp = 'donkey_60.jpg';
%img_name = regexprep(char(names_test(1)), '\.jpg', '');
%img_fn = fullfile(img_dir, char(names_test(1)));
img_name = regexprep(temp, '\.jpg', '');
img_fn = fullfile(img_dir, temp);
svm = struct;
svm.supVec = supVec;
svm.wVec = wVec;
svm.bVec = bVec;
svm.kernel = kernel;
svm.kerneloption = kerneloption;
[img, ucm2, mask2] = gPb(img_fn, 'out/ayahoo_test_images/processed');
hierarchy(img, img_name, svm, mask2, ucm2, 1, '');
end