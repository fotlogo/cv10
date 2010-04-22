%-----------------------------------------------------------
% Experimental function recursing through hierarchy of
% segmentations.
%
% Definitions:
% ucm - Ultrametric Contour Map
% ucm2_orig - The original ucm2
% mask - a binary mask with ones anywhere the child region
%        is defined
% xxx2 - Some map xxx at twice the size of the original 
%        image
%
% Some output ideas:
%
% bdry = (ucm >= k);
% figure; imshow(bdry);
%
% figure; imshow(ucm);
% figure; imshow(labels); colormap(jet);
% figure;imshow(mask,[]);colormap(jet);
%
%-----------------------------------------------------------
function [] = hierarchy(img, features, mask2, ucm2_orig, depth, path, visit)

%---------------------------------------
% For testing, return at a maximum
% depth
%---------------------------------------
if depth > 5,
  return;
end

% Mask out original size ucm
mask = mask2(3:2:end, 3:2:end);
ucm2 = ucm2_orig;
ucm = ucm2(3:2:end, 3:2:end);
ucm = bsxfun(@times, ucm, uint8(mask));

% Get k from masked-out original size ucm
k = max(max(ucm)) - 1;

% Use k to label double-sized ucm
labels2 = bwlabel(ucm2_orig <= k);

% Get rid of '0' labels (by changing them to 
% a value one greater than the greatest label).
l = max(max(labels2));
labels2(find(labels2==0)) = l+1;

% Mask-out pixels in labeling by changing them to 0.
labels2 = bsxfun(@times, labels2, double(mask2));
labels = labels2(2:2:end, 2:2:end);

% Get the vector of unique labels from the original
% size labeling and remove '0' entry of unique labels
% since those regions are masked out.
label_vec = unique(labels);
label_vec(label_vec == 0) = [];

% debug output
disp(sprintf('%s attributes: %d, k: %d, num_labels: %d, [%d %d]', ...
	     path, classify(ucm), k, size(label_vec, 1), min(label_vec), ...
	     max(label_vec)));

cm = jet(min(256, max(unique(labels2))));
imwrite(bsxfun(@times, img, uint8(mask)), sprintf('output/img%s.jpg', path));
%imwrite(ucm2, sprintf('output/ucm%s.bmp', path));
%imwrite(labels2, cm, sprintf('output/labels%s.bmp', path));
%imwrite(labels, cm, sprintf('output/labels_%s.bmp', path));
%imwrite(mask2, sprintf('output/mask%s.bmp', path));
%imwrite(ucm2_orig<=k, sprintf('output/logical%s.bmp', path));

if (visit)
  visit(img, features, mask, ucm);
end

if (size(label_vec, 1) > 1),
  for i=1:size(label_vec, 1),
    sub_mask2 = (labels2 == label_vec(i));
    sub_mask2 = bsxfun(@times, sub_mask2, mask2);
    hierarchy(img, features, sub_mask2, ucm2_orig, depth+1,...
	      sprintf('%s%d', path, i), visit);
  end
end
