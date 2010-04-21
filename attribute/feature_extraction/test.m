function [] = test()
%addpath('attribute/feature_extraction');
%addpath('attribute/feature_extraction/code');
%addpath('attribute/feature_extraction/code/textons');
%cd 'attribute/feature_extraction';
%extract_features_dh('../../data', '../../out');
%cd ../..;

addpath('code');
addpath('code/textons');
extract_features_dh('data', 'out');
