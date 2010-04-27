function [attPred,attConf] = visitor(img, img_name, svm, mask, ucm, ...
				     path)
global num_atts atts atts_mask;

img_masked = bsxfun(@times, img, uint8(mask));
[features]  = get_features(img_name, mask);

%figure,imshow(double(mask))

if (nnz(features) == 0)
  disp('No features at all!  Why is this working?');
end

%---------------------------------------
% See what attributes we get for the 
% image
%---------------------------------------
att_pred = zeros(1, num_atts);
for i = 1:num_atts
  y=svmval(features, svm.supVec{i}, svm.wVec{i}, svm.bVec{i}, ...
	   svm.kernel, svm.kerneloption);
  %y(y>0)=1;
  %y(y<=0)=0;
  att_pred(:,i)=y;
  %disp(sprintf('%2d: positive = %3d; precision = %1.2f', i, sum(y==1), ...
  %  sum(y==att_actual(:,i))/length(y)));
end

attConf=att_pred;

att_pred=att_pred.*atts_mask';
att_pred(att_pred<=0)=0;

attPred=(att_pred>0);

[sort_value,sort_idx]=sort(att_pred,'descend');
att_sort=sort_idx(sort_value>0);

%st = sprintf('%f', att_pred);
%disp(['atts: ' st]);
%att_pred
%sort_value
atts(att_sort)
sort_value(sort_value>0)

base_name = regexprep(char(img_name), '.jpg', '');
fid = fopen(sprintf('out/%s/atts%s.txt', base_name, path), 'w');
atts_sorted = atts(att_sort);
sort_value = sort_value(sort_value>0);
for i=1:size(att_sort(:))
  fprintf(fid, '%1.3f %s\n', sort_value(i), atts_sorted{i});
end
fclose(fid);

