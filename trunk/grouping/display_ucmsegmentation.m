function bdry = display_ucmsegmentation(imgFile, k)
%% usage example

%clear all;close all;clc;
close all;

%read double sized ucm
%ucm2 = imread('data/101087_ucm2.bmp');
ucm2 = imread([imgFile '_ucm2.bmp']);

% convert ucm to the size of the original image
ucm = ucm2(3:2:end, 3:2:end);

% get the boundaries of segmentation at scale k in range [1 255]
%k = 75;
bdry = (ucm >= k);

% get the partition at scale k without boundaries:
labels2 = bwlabel(ucm2 <= k);
labels = labels2(2:2:end, 2:2:end);

%figure;imshow('data/101087.jpg');
figure;imshow(imgFile);
figure;imshow(ucm);
figure;imshow(bdry);
figure;imshow(labels,[]);colormap(jet);

pause;
close all;