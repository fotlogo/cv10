function [] = extract()
cd '~edwardsj/classes/cs395T_vision/project/cv10';

%---------------------------------------
% Process features
%---------------------------------------
%addpath('attribute/feature_extraction');
%addpath('attribute/feature_extraction/code');
%addpath('attribute/feature_extraction/code/textons');
cd 'attribute/feature_extraction';

addpath('code');
addpath('code/textons');

% aYahoo
%extract_features_dh('../../data/ayahoo_test_images', '../../out/ayahoo_test_images');

%PASCAL 2008
extract_features_dh('../../data/apascal_images', '../../out/apascal_images');

cd ../..;

%exit