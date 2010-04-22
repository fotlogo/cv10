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
addpath('lib')

gPb = 0;

%% compute globalPb
if (gPb),
  clear all; close all; clc;

  imgFile = 'data/101087.jpg';
  outFile = 'data/101087_gPb.mat';
  rsz = 0.5;

  globalPb(imgFile, outFile, rsz);

  %% compute Hierarchical Regions
  clear all; close all; clc;

  load data/101087_gPb.mat gPb_orient

  % for boundaries
  ucm = contours2ucm(gPb_orient, 'imageSize');
  imwrite(ucm,'data/101087_ucm.bmp');

  % for regions
  ucm2 = contours2ucm(gPb_orient, 'doubleSize');
  imwrite(ucm2,'data/101087_ucm2.bmp');

end

clear all;close all;clc;

%% usage example

%read double sized ucm
ucm2 = imread('data/101087_ucm2.bmp');

% convert ucm to the size of the original image
ucm = ucm2(3:2:end, 3:2:end);
orig_img = imread('data/101087.jpg');

% get the boundaries of segmentation at scale k in range [1 255]
mask2 = ones(size(ucm2,1), size(ucm2,2));
hierarchy(orig_img, 0, mask2, ucm2, 1, '1', 0);
return

k = 100;
bdry = (ucm >= k);

%figure; imshow((ucm >= 241));
%figure; imshow((ucm >= 236));

%unique(ucm)

% get the partition at scale k without boundaries:
labels2 = bwlabel(ucm2 <= k);
labels = labels2(2:2:end, 2:2:end);

imwrite(labels2, colormap(jet), sprintf('output/labels.bmp'));
imwrite(labels2, sprintf('output/bdry.bmp'));


indices = randperm(size(unique(labels), 1));
for i=1:min(0, size(indices,2)),
%  region = uint8((labels == median(unique(labels))));
  region = uint8((labels == indices(i)));
  img = orig_img;
  img = bsxfun(@times, img, region);
  %figure;imshow(img);
  imwrite(labels2, sprintf('output/labels%d.bmp', i));
end

%imwrite(labels,'data/101087_labels.bmp');
%imwrite(region,'data/101087_region.bmp');
%imwrite(bdry,'data/101087_bdry.bmp');

%figure;imshow('data/101087.jpg');
%figure;imshow(ucm);
%figure;imshow(bdry);
%figure;imshow(labels,[]);colormap(jet);

