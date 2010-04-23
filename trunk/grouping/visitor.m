function [] = visitor(img, img_name, svm, mask, ucm)
global num_atts atts;

img_masked = bsxfun(@times, img, uint8(mask));
[features]  = get_features(img_name, mask);
%disp(sprintf('%f', features(1:20:end, 1:20:end)'));

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

att_pred(att_pred<=0)=0;
[sort_value,sort_idx]=sort(att_pred,'descend');
att_sort=sort_idx(sort_value>0);

%st = sprintf('%f', att_pred);
%disp(['atts: ' st]);
%att_pred
%sort_value
atts(att_sort)

