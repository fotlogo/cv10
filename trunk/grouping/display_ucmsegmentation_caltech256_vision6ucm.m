function [img bdry] = display_ucmsegmentation_caltech256_vision6ucm(classID, imageID, k, disp_flag)
%% usage example

baseDir = '/projects/vision/6/jaechul/caltech256/256_ObjectCategories/';
ucmbaseDir = '/projects/vision/6/jaechul/caltech256/256_ObjectCategories/';

className = dir([baseDir classID '.*']);

classDir = [baseDir className.name];
ucmclassDir = [ucmbaseDir className.name];

imgPrefix = '0000';
imgPrefix(end-length(num2str(imageID))+1:end) = num2str(imageID);
imgFile = [classDir '/' classID '_' imgPrefix '.jpg'];

ucmFile = [ucmclassDir '/' classID '_' imgPrefix '.jpg'];
%clear all;close all;clc;
close all;

%read original image
img = imread(imgFile);
%read double sized ucm
%ucm2 = imread('data/101087_ucm2.bmp');
ucm2 = imread([ucmFile '_ucm2.bmp']);

% convert ucm to the size of the original image
ucm = ucm2(3:2:end, 3:2:end);
max_edge_strength = max(ucm(:));
non_zero_response = ucm(ucm > 10.0);
mean_edge_strength = mean(non_zero_response(:));

% get the boundaries of segmentation at scale k in range [1 255]
%k = 75;
bdry = (ucm >= k);
% get the partition at scale k without boundaries:
labels2 = bwlabel(ucm2 <= k);
labels = labels2(2:2:end, 2:2:end);

if ( disp_flag == 1 )
    %figure;imshow('data/101087.jpg');
    figure;imshow(imgFile);
    figure;imshow(ucm);
    figure;imshow(bdry);
    figure;imshow(labels,[]);colormap(jet);
    disp(num2str(max_edge_strength));
    disp(num2str(mean_edge_strength));
    
    pause;
    close all;
end
