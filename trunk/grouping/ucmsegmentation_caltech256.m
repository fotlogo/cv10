% DESCRIPTION:
%   Code to compute globalPb and hierarchical regions, as described in:
%
%   P. Arbelaez, M. Maire, C. Fowlkes and J. Malik. 
%   From Contours to Regions: An Empirical Evaluation. CVPR 2009.
%
%   M. Maire, P. Arbelaez, C. Fowlkes and J. Malik. 
%   Using Contours to Detect and Localize Junctions in Natural Images. CVPR
%   2008.
% 
% WARNINGS:
%   This code is still under development and testing. It is being distributed on its
%   present form for educational and research purposes only. The final public
%   release will probably be different. 
%
%   If you use any portion of this code, please acknowledge our work by
%   citing the two papers above.
%
%   Please report any bugs or improvements to the address below.
%
%   latest version : April 1st 2009
%
%   Pablo Arbelaez.
%   <arbelaez@eecs.berkeley.edu>

%% DIRECTIONS: 
%  unzip and update the absolute path in the file lib/spectralPb.m
% 
%%
function ucmsegmentation_caltech256(classID, imageID, rsz)

baseDir = '/projects/vision/6/jaechul/caltech256/256_ObjectCategories/';
dstDir = '/projects/vision/9/jaechul/caltech256/256_ObjectCategories/';

classDirInfo = dir([baseDir classID '.*']);
classDir = [baseDir classDirInfo.name];
outDir = [dstDir classDirInfo.name];

imgPrefix = '0000';
imgPrefix(end-length(num2str(imageID))+1:end) = num2str(imageID);
imgFile = [classDir '/' classID '_' imgPrefix '.jpg'];

outFile = [outDir '/' classID '_' imgPrefix '_ucm.mat'];
outImgFile = [outDir '/' classID '_' imgPrefix '.jpg'];
% if ( ~isempty(dir(outFile)) )
%     return;
% end
%addpath('/projects/vision/6/jaechul/h-seg/grouping/lib')
addpath('lib')

%% compute globalPb
%clear all; close all; clc;
close all; clc;
%imgFile = 'data/101087.jpg';
%outFile = 'data/101087_gPb.mat';
%rsz = 0.5;
globalPb(imgFile, outFile, rsz);

%% compute Hierarchical Regions
%clear all; close all; clc;

%load data/101087_gPb.mat gPb_orient
load(outFile, 'gPb_orient');

% for boundaries
ucm = contours2ucm(gPb_orient, 'imageSize');
%imwrite(ucm,'data/101087_ucm.bmp');
imwrite(ucm, [outImgFile '_ucm.bmp']);
% for regions
ucm2 = contours2ucm(gPb_orient, 'doubleSize');
%imwrite(ucm2,'data/101087_ucm2.bmp');
imwrite(ucm2,[outImgFile '_ucm2.bmp']);

%% usage example

%clear all;close all;clc;

%read double sized ucm
%ucm2 = imread('data/101087_ucm2.bmp');
%ucm2 = imread([imgFile '_ucm2.bmp']);

% convert ucm to the size of the original image
%ucm = ucm2(3:2:end, 3:2:end);

% get the boundaries of segmentation at scale k in range [1 255]
%k = 100;
%bdry = (ucm >= k);

% get the partition at scale k without boundaries:
%labels2 = bwlabel(ucm2 <= k);
%labels = labels2(2:2:end, 2:2:end);

%figure;imshow('data/101087.jpg');
%figure;imshow(imgFile);
%figure;imshow(ucm);
%figure;imshow(bdry);
%figure;imshow(labels,[]);colormap(jet);

