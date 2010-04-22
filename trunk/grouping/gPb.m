%% compute globalPb
function [img, ucm2, mask2] = gPb(fn, outdir)

%clear all; close all; clc;

[pathstr, name, ext, versn] = fileparts(fn);

imgFile = fn; %'data/101087.jpg';
outFile = sprintf('%s/%s_gPb.mat', outdir, name);
outucm = sprintf('%s/%s_ucm.bmp', outdir, name);
outucm2 = sprintf('%s/%s_ucm2.bmp', outdir, name);

if ~exist(outFile, 'file') 

  rsz = 0.5;

  globalPb(imgFile, outFile, rsz);

  %% compute Hierarchical Regions
  %clear all; close all; clc;

  %load data/101087_gPb.mat gPb_orient
  load(outFile, 'gPb_orient');

  % for boundaries
  ucm = contours2ucm(gPb_orient, 'imageSize');
  %imwrite(ucm,'data/101087_ucm.bmp');
  imwrite(ucm, outucm);

  % for regions
  ucm2 = contours2ucm(gPb_orient, 'doubleSize');
  %imwrite(ucm2,'data/101087_ucm2.bmp');
  imwrite(ucm2, outucm2);
end

ucm2 = imread('data/101087_ucm2.bmp');
img = imread('data/101087.jpg');
mask2 = ones(size(ucm2,1), size(ucm2,2));
