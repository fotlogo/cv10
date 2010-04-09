function display_ucmsegmentation_caltech256_vision9ucm(classID, imageID)
%% usage example

baseDir = '/projects/vision/6/jaechul/caltech256/256_ObjectCategories/';
ucmbaseDir = '/projects/vision/9/jaechul/caltech256/256_ObjectCategories/';

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
non_zero_response = ucm(ucm > 10.0);
mean_edge_strength = mean(non_zero_response(:));
disp(mean_edge_strength);
% get the boundaries of segmentation at scale k in range [1 255]
%k = 75;
longerD = max([size(img,1), size(img,2)]);
rsz = 1;
if ( longerD > 320 )
    rsz = 320/longerD;
end
disp(['resize by ' num2str(rsz)]);
img = imresize(img, rsz);
subplot(2, 4, 1); imshow(img);
k_count = 1;
for k = 1.0 : 0.25 : 2.5
    %bdry = (ucm >= k*mean_edge_strength);
    %se = strel('disk', 1);
    %dilated_bdry = imdilate(bdry, se);
    
    %dilated_bdry = imresize(dilated_bdry, rsz);
    % get the partition at scale k without boundaries:
    labels =  bwlabel(ucm <= k*mean_edge_strength);
    disp(max(labels(:)));
    labels = imresize(labels, rsz, 'nearest');
    
    subplot(2, 4, k_count+1); imshow(label2rgb(labels));
    k_count = k_count + 1;
end
%     imR = img(:,:,1);
%     imG = img(:,:,2);
%     imB = img(:,:,3);
%     imR(dilated_bdry==1) = 255;
%     imG(dilated_bdry==1) = 0;
%     imB(dilated_bdry==1) = 0;
%     img(:,:,1) = imR;
%     img(:,:,2) = imG;
%     img(:,:,3) = imB;

% if ( disp_flag == 1 )
%     
%     %figure;imshow('data/101087.jpg');
%     figure;imshow(img);
%     %figure;imshow(ucm,[]);
%     %figure; imshow(bdry,[]);
%     %hold on; imshow((dilated_bdry==1),[]);
%     %figure;imshow(labels,[]);colormap(jet);
%     disp(num2str(max_edge_strength));
%     disp(num2str(mean_edge_strength));
%     
%     pause;
%     close all;
% end
