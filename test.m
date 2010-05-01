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
%   * The variable 'num_atts' controls how many attributes are
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

%function [] = test(depth)

%if (nargin < 1)
%  depth = 2;
%end

% clear

%BasePath='/u/atian/cv/final/code/';
%BasePath='/u/edwardsj/classes/cs395T_vision/project/cv10/';
BasePath=getenv('VISION_BASE_PATH');

addpath(BasePath);
addpath([BasePath,'grouping']);
addpath([BasePath,'grouping/lib']);
addpath([BasePath,'libsvm'])
addpath([BasePath,'SVM-KM'])

global img_dir hog_dir tc_dir num_atts atts atts_mask kernel kerneloption C verbose lambda;


%image_set = 'ayahoo_test';
image_set = 'apascal';


gPbdir = [BasePath,'out/',image_set,'_images/processed/gPb'];

%---------------------------------------
% get attributes and bounding boxes for
% data
%---------------------------------------
%fname = [BasePath,'data/attribute_data/',image_set,'.txt'];
fname_train = [BasePath,'data/attribute_data/',image_set,'_train.txt'];
fname_test = [BasePath,'data/attribute_data/',image_set,'_test.txt'];
img_dir = [BasePath,'data/',image_set,'_images'];
hog_dir = [BasePath,'out/',image_set,'_images/processed/hog'];
tc_dir = [BasePath,'out/',image_set,'_images/processed/tc2'];

[img_names img_classes bboxes attributes] = read_att_data(fname_train);
names_train = img_names;
classes_train = img_classes;
bboxes_train = bboxes;
attributes_train = attributes;
count_train = size(names_train(:));

[img_names img_classes bboxes attributes] = read_att_data(fname_test);
names_test = img_names;
classes_test = img_classes;
bboxes_test = bboxes;
attributes_test = attributes;
count_test = size(names_test(:));

TRAIN = 0;
TEST = 0;
SEGMENTATION = 1;
FEATURE_TRAIN=0;
FEATURE_TEST=0;
%count_train = 2000;
%count_test = 500;

%% change to random of permutation here, Aibo
%rand('seed', 1);
%
ratio=0.5  % ratio of positive samples for training
att_train_thre=50; % lower threshold of training samples for each attribute
%
%rand_indices=randperm(length(img_names));
%train_indices=rand_indices(1:count_train);
%test_indices=rand_indices(count_train+1:count_train+count_test);

%---------------------------------------
% split image names and classes up into
% training and testing sets
%---------------------------------------

%names_train = img_names(train_indices,:);
%names_test = img_names(test_indices,:);
%classes_train = img_classes(train_indices,:);
%classes_test = img_classes(test_indices,:);
%bboxes_train = bboxes(train_indices,:);
%bboxes_test = bboxes(test_indices,:);
%attributes_train = attributes(train_indices,:);
%attributes_test = attributes(test_indices,:);

% count_train = size(names_train, 1);
% count_test = size(names_test, 1);

%---------------------------------------
% get the attribute names
%---------------------------------------
fid = fopen([BasePath,'data/attribute_data/attribute_names.txt']);
[atts] = textscan(fid, '%s', 'delimiter', '\n');
atts = atts{1};
fclose(fid);

%---------------------------------------
% get the class names
%---------------------------------------
fid = fopen([BasePath,'data/attribute_data/class_names.txt']);
[classes] = textscan(fid, '%s', 'delimiter', '\n');
classes = classes{1};
fclose(fid);

%---------------------------------------
% Load all of the features for the
% dataset
%---------------------------------------
num_atts = size(atts, 1);

% attributes mask
atts_mask=ones(size(atts));

%features_train = zeros(count_train, num_features);
features_train = [];
%labels_train = zeros(count_train, 1);


% SVM KM Kernel Parameters
% -------------------------------------------------------
kernel='gaussian';
kerneloption=1
C=100000000;
verbose=0;
lambda=1e-7;
%nbclass=2;


%---------------------------------------
% train a classifier for each attribute
%---------------------------------------
if (TRAIN==1)
    disp ('Start training................')

    % get the features for training images
    if FEATURE_TRAIN==1
        for i = 1:count_train
            img_name = regexprep(char(names_train(i)), '\.jpg', '');
            [feat]  = get_features(img_name, bboxes_train(i,:));
            %num_features = size(feat, 2);
	    disp(img_name);
            features_train = [features_train; feat];

            %labels_train(i,:) = find(strcmp(classes, classes_train{i}));
            %disp(sprintf('%d %s %s', i, classes_train{i}, img_name));
        end
        save([BasePath,'features_train'],'features_train')
        disp('Extract features done...')
    else
        load([BasePath,'features_train'])
        disp('Load features done...')
    end
    num_features=size(features_train,2);
    
    % remove bad features
    temp1=sum(features_train,2);
    temp2=(temp1>=0);
    features_train=features_train(temp2,:);
    attributes_train=attributes_train(temp2,:);

  % train the classifier
  %att_pred = zeros(count_train, num_atts);
  
  % svm km
  % -------------------------------------------------------
  % Solving
  % -------------------------------------------------------
  ypred=[];
  supVec={};
  wVec={};
  bVec={};
  
  for i = 1:num_atts
    fprintf('training attribute %d\n', i);
    att = attributes_train(:,i);
    att_pos=find(att==1);
    
    rand('seed',1);

    if length(att_pos)>500
        rand_temp=randperm(length(att_pos));
        att_pos=att_pos(rand_temp(1:500));
    end
    
    % change attributes mask
    %if length(att_pos)<att_train_thre
    %  atts_mask(i)=0;
    %end

    att_neg=find(att==0);
    if ((length(att_neg)/length(att_pos))>((1-ratio)/ratio))
        rand_temp=randperm(length(att_neg));
        att_neg=att_neg(rand_temp(1:floor(length(att_pos)/ratio*(1-ratio))));
    end
    if length(att_pos)==0
        features_temp=features_train;
        att_temp=att;
        num_pos=0;
    else
        features_temp=[features_train(att_pos,:);features_train(att_neg,:)];
        att_temp=[att(att_pos,:);att(att_neg,:)];
        num_pos=length(att_pos);
    end
    
    % svm km
    att_temp(att_temp==0)=-1;
    [xsup,w,b,pos,timeps,alpha,obj]=svmclass(features_temp,att_temp,C,lambda,kernel,kerneloption,verbose);
    supVec=[supVec,xsup];
    wVec=[wVec,w];
    bVec=[bVec,b];
    y=svmval(features_temp,xsup,w,b,kernel,kerneloption);
    y(y>0)=1;
    y(y<=0)=0;
    %y(isnan(y))=0;
    
    % disp precision
    att_temp(att_temp==-1)=0;
    precision=sum(y==att_temp)/length(att_temp);
    precisionPos=y'*att_temp/sum(att_temp);
    att_temp=double(~att_temp);
    y_temp=double(~y);
    precisionNeg=y_temp'*att_temp/sum(att_temp);
    disp(['#pos ',num2str(num_pos),' #total ',num2str(length(att_temp)),' precision ',num2str(precision),' pos ',num2str(precisionPos),' neg ',num2str(precisionNeg)])

    fprintf('.')
  end
  fprintf('\n')
  %save([BasePath,'models.mat'], 'supVec', 'wVec', 'bVec','atts_mask');
  disp('SVM training done...')
%disp('Predicted attributes');
  %for i = 1:size(att_pred, 1)
  %  disp(sprintf('%u', att_pred(i,:)));
  %end
  %save('models.mat', 'models');
  %save('models_small.mat', 'models');

else
  % if we're not training, load the classifiers from disk
  disp(sprintf('SVM loading models.mat...'));
  load([BasePath,'models.mat']);
  %load('models_small.mat');
end

atts_mask=logical(atts_mask);

% output attributes
fprintf('atts_mask = %s\n', sprintf('%u', atts_mask'));
disp(atts(logical(atts_mask)));

%---------------------------------------
% test the classifiers
%---------------------------------------
if (TEST==1)
  disp ('Start testing ..................')

  %---------------------------------------
  % get features for test images
  %---------------------------------------
  %features_test = zeros(count_test, num_features);
  features_test=[];
  %labels_test = zeros(count_test, 1);

  % get the features for test images
  if FEATURE_TEST==1
        for i = 1:count_test
            img_name = regexprep(char(names_test(i)), '\.jpg', '');
            [feat]  = get_features(img_name, bboxes_test(i,:));
            features_test(i,:) = feat;
            %labels_test(i,:) = find(strcmp(classes, classes_test{i}));
            %disp(sprintf('%d %s %s', i, classes_test{i}, img_name));
        end
        save([BasePath,'features_test'],'features_test')
        disp('Extract features done...')
  else
      load([BasePath,'features_test'])
      disp('Load features done...')
  end
  num_features=size(features_test,2);
  
    % remove bad features
    temp1=sum(features_test,2);
    temp2=(temp1>=0);
    features_test=features_test(temp2,:);
    attributes_test=attributes_test(temp2,:);


  %---------------------------------------
  % See what attributes we get for the 
  % test images
  %---------------------------------------
  att_pred = zeros(count_test, num_atts);
  att_actual = attributes_test;
  features = features_test;
  precision=[];
  total_precision=[];
  total_pos_precision=[];
  total_neg_precision=[];
  for i = 1:num_atts
    num_pos=sum(att_actual(:,i)>0);
    % svm km
    y=svmval(features,supVec{i},wVec{i},bVec{i},kernel,kerneloption);
%     y(y>0)=1;
%     y(y<=0)=0;
%     att_pred(:,i)=y;
%     disp(sprintf('%2d: positive = %3d; precision = %1.2f', i, sum(y==1), ...
% 		 sum(y==att_actual(:,i))/length(y)));
%     precision=[precision,sum(y==att_actual(:,i))/length(y)];

    y(y>0)=1;
    y(y<=0)=0;
    % disp precision
    att_temp=att_actual(:,i);
    % total precision
    precision=sum(y==att_temp)/length(att_temp);
    total_precision=[total_precision,precision];
    % pos precision
    if sum(att_temp)==0 % no positive
        precisionPos=-1;
    else
        precisionPos=y'*att_temp/sum(att_temp);
    end
    total_pos_precision=[total_pos_precision,precisionPos];
    % neg precision
    att_temp=double(~att_temp);
    y_temp=double(~y);
    if sum(att_temp)==0 % no negative
        precisionNeg=-1;
    else
        precisionNeg=y_temp'*att_temp/sum(att_temp);
    end
    total_neg_precision=[total_neg_precision,precisionNeg];
    
    disp(['#pos ',num2str(num_pos),' #total ',num2str(length(att_temp)),' precision ',num2str(precision),' pos ',num2str(precisionPos),' neg ',num2str(precisionNeg)])

  end
  disp(['Total average precision ',num2str(mean(total_precision))])
  disp(['Total pos average precision ',num2str(mean(total_pos_precision(total_pos_precision>=0)))])
  disp(['Total neg average precision ',num2str(mean(total_neg_precision(total_neg_precision>=0)))])
  disp(['Total average precision after mask ',num2str(mean(total_precision(logical(atts_mask))))])
  disp(['Total pos average precision after mask ',num2str(mean(total_pos_precision(logical((total_pos_precision>=0).*atts_mask'))))])
  disp(['Total neg average precision after mask ',num2str(mean(total_neg_precision(logical((total_neg_precision>=0).*atts_mask'))))])

  
  %st = arrayfun(@(x)sprintf('%u', x), att_pred);
  %disp(st);
  %disp(atts);
%   disp(sprintf('total precision: %1.2f', sum(precision)/length(precision)));
end



if (SEGMENTATION)

  % images ~5K size
  %temp = 'donkey_60.jpg';
  %temp = 'jetski_158.jpg';
  
  % images ~10K size
  %temp = 'mug_248.jpg';
  %temp = 'goat_361.jpg';
  
  % images ~20K size
  %temp = 'bag_377.jpg';
  %temp = 'monkey_220.jpg'; 
  %temp = 'goat_12.jpg';
  %temp = 'bag_377.jpg';

  %image_set = 'ayahoo_test';
  image_set = 'apascal';
  gPbdir = [BasePath,'out/',image_set,'_images/processed/gPb'];
  img_dir = [BasePath,'data/',image_set,'_images'];
  hog_dir = [BasePath,'out/',image_set,'_images/processed/hog'];
  tc_dir = [BasePath,'out/',image_set,'_images/processed/tc2'];

  %temp = 'monkey_220.jpg';
  %temp = 'monkey_221.jpg';
  temp='2008_007214.jpg';
  
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
  [img, ucm2, mask2] = gPb(img_fn, sprintf('out/%s_images/processed',image_set));

  base_name = regexprep(char(temp), '.jpg', '');
  out_dir = [BasePath,'out/',base_name];
  if (exist(out_dir, 'dir') ~= 7)
    mkdir(out_dir);
  end
  
% attStruct is the structure with attributes hierarchy
  %attStruct=hierarchy(img, img_name, svm, mask2, ucm2, depth, '', @visitor);
  
  load(['/u/atian/attStruct.mat']);
  attStruct2=hierarchyMod(img, img_name, svm, mask2, ucm2, depth, '', attStruct);

  %img_name = regexprep(char(temp), 'jpg', 'mat');
  %save([out_dir,'/attStruct'], 'attStruct');

  % TODO
  % define the pattern of car model
  car_pattern=[1,2,25,27,28,29,30,31,32,53,62];
  car_model=zeros(1,num_atts);
  car_model(car_pattern)=1;
  % give some threshold of attribute probability
  att_thre=0.5;
  model_sim_thre=0.4;
  model_sim_min=0.1;
  % calculate histogram intersection
  % find the node first then prune
  %attStruct.obj=0;
  obj_seg=[];
  [attStructPrune,obj_seg]=pruneHier(attStruct,att_thre,model_sim_thre,model_sim_min,car_model,obj_seg,0);
  
  img=imread([img_dir,'/',img_name,'.jpg']);
  img(:,:,1)=img(:,:,1).*uint8(obj_seg.mask);
  img(:,:,2)=img(:,:,2).*uint8(obj_seg.mask);
  img(:,:,3)=img(:,:,3).*uint8(obj_seg.mask);
  figure,imshow(img)
  
  
end
