function [] = visitor(img, img_name, svm, mask, ucm)
global num_atts;

img_masked = bsxfun(@times, img, uint8(mask));
[features]  = get_features(img_name, mask);

%---------------------------------------
% See what attributes we get for the 
% image
%---------------------------------------
att_pred = zeros(1, num_atts);
for i = 1:num_atts
  y=svmval(features, svm.supVec{i}, svm.wVec{i}, svm.bVec{i}, ...
	   svm.kernel, svm.kerneloption);
  y(y>0)=1;
  y(y<=0)=0;
  att_pred(:,i)=y;
  %disp(sprintf('%2d: positive = %3d; precision = %1.2f', i, sum(y==1), ...
  %  sum(y==att_actual(:,i))/length(y)));
end
disp(sprintf('%u', att_pred));


