function display_images(fname, imdir)
% Example function for using read_att_data.m, and the dataset
% Input: <fname> name of dataset to use
%        <imdir> directory where images are stored

[img_names img_classes bboxes attributes] = read_att_data(fname);

fid = fopen('data/attribute_data/attribute_names.txt');
[atts] = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);

atts = atts{1};

for i = 1:100:length(img_names)
   im = imread(fullfile(imdir, img_names{i}));

   bbox = bboxes(i,:);
   disp(sprintf('%s', img_classes{i}));
   atts(find(attributes(i,:)))'
%   subplot(1,2,1)
%   imagesc(im);
%   subplot(1,2,2)
%   imagesc(im(bbox(2):bbox(4), bbox(1):bbox(3), :))
%   title(img_classes{i})
%   pause
end
