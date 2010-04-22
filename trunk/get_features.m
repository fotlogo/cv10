function [feat] = get_features(img_dir, img_name, ...
					hog_dir, tc_dir,...
					mask, num_features)

im = imread(fullfile(img_dir, sprintf('%s.jpg', img_name)));

% [x_min y_min x_max y_max]
%box = bboxes(i,:);

hogfn = fullfile(hog_dir, sprintf('%s_hog.mat', img_name));
hog = load(hogfn);

tcfn = fullfile(tc_dir, sprintf('%s_tc.mat', img_name));
tc = load(tcfn);

if (size(mask, 2) == 4)
  box = mask;
  hog_ind = hog.x >= box(1) & hog.y >= box(2) & hog.x <= box(3) & ...
	    hog.y <= box(4);

  feat_hog = hist(hog.idx(hog_ind),1:1000);
  feat_hog_norm = feat_hog/norm(feat_hog);

  %feat_color = hist(tc.colorim(box(2):box(4), box(1):box(3)),1:128);
  %feat_color_norm = feat_color/norm(feat_color);

  %feat_texture = hist(tc.textonim(box(2):box(4), box(1):box(3)),1:256);
  %feat_texture_norm = feat_texture/norm(feat_texture);

else
  hog_ind = mask(hog.y, hog.x)
end

%size(feat_hog_norm)
%size(feat_color_norm)
%size(feat_texture_norm)

feat = feat_hog_norm; %[feat_hog_norm feat_color_norm feat_texture_norm];
