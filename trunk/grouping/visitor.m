function [] = visitor(img, features, ucm, mask)

img_masked = bsxfun(@times, img, uint8(mask));
% mask features - not implemented

%---------------------------------------
% See what attributes we get for the 
% image
%---------------------------------------
att_pred = zeros(1, numatts);
for i = 1:64
  [T, predict_label, accuracy, dec_values] = evalc('svmpredict(att_actual(i), features, models(i))');
  att_pred(i) = predict_label;
end
disp(sprintf('%u', att_pred));


