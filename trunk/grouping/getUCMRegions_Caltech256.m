function [img regions unique_subregions contour_map] = getUCMRegions_Caltech256(classID, imageID, max_contour_samples, sample_rate)

close all;
baseDir = '/projects/vision/6/jaechul/caltech256/256_ObjectCategories/';
ucmbaseDir = '/projects/vision/9/jaechul/caltech256/256_ObjectCategories/';

className = dir([baseDir classID '.*']);

classDir = [baseDir className.name];
ucmclassDir = [ucmbaseDir className.name];

imgPrefix = '0000';
imgPrefix(end-length(num2str(imageID))+1:end) = num2str(imageID);
imgFile = [classDir '/' classID '_' imgPrefix '.jpg'];

ucmFile = [ucmclassDir '/' classID '_' imgPrefix '.jpg'];

%read original image
img = imread(imgFile);
%read double sized ucm
%ucm2 = imread('data/101087_ucm2.bmp');i
ucm2 = imread([ucmFile '_ucm2.bmp']);
% convert ucm to the size of the original image
ucm = ucm2(3:2:end, 3:2:end);

% adaptive threshold for segmentation
non_zero_response = ucm(ucm > 10.0);
mean_edge_strength = mean(non_zero_response(:));

% resize factor
longerD = max(size(img));
rsz = 1;
if ( longerD > 320 )
    rsz = 320/longerD;
end
img = imresize(img, rsz);
region_count = 1;
% get the segmentation
se = strel('disk', 1); % for dilation
contour_map = false(size(img,1), size(img,2));

for k = 0.5 : 0.25 : 2.0
    threshold = k*mean_edge_strength;
    labels =  bwlabel(ucm <= threshold);
    
    ori_contours = labels == 0;
    ori_contours = imdilate(ori_contours,se);
    ori_contours = imresize(ori_contours, rsz, 'nearest');
    labels = imresize(labels, rsz, 'nearest');
    
    nlabels = max(labels(:));
    if ( nlabels > 50 ) % too detailed segmentation is rejected.
        continue;
    end
    %ori_contours = labels == 0;
    region_property = regionprops(double(labels), 'Area', 'Centroid', 'BoundingBox', 'Perimeter', 'PixelIdxList', 'PixelList');
    for i = 1 : max(labels(:)) % labels == 0 : contour
        if ( region_property(i).Perimeter >= 24*4) % small regions are excluded
            region = bwlabel(labels == i);
            contour = bwperim(region) == 1;
            contour(:,1) = 0;
            contour(:,end) =0;
            contour(1,:) = 0;
            contour(end,:) = 0;
            contour = imdilate(contour,se);
            contour = contour & ori_contours;
            contour_map = contour_map | contour;
            regions(region_count).interior = region_property(i).PixelIdxList;
            regions(region_count).contour = find(contour);
            w = ceil(region_property(i).BoundingBox(3)) + 1;
            h = ceil(region_property(i).BoundingBox(4)) + 1;
            mask_ones =...
                region_property(i).PixelList - repmat(floor(region_property(i).BoundingBox(1:2)), region_property(i).Area, 1) + 1;
            mask_ones = sub2ind([h,w], mask_ones(:,2), mask_ones(:,1));
            regions(region_count).mask = false(h,w);
            regions(region_count).mask(mask_ones) = 1;
            regions(region_count).area = region_property(i).Area;
            regions(region_count).centroid = region_property(i).Centroid;
            regions(region_count).boundingbox = region_property(i).BoundingBox;
            regions(region_count).perimeter = region_property(i).Perimeter;
            regions(region_count).imageSize = size(img); %used for converting linear indices (e.g., contour and interior) to subscripts
            region_count = region_count + 1;
        end        
    end
end
% identify the same regions and merge them into one
areas = [regions.area]';
centroids = reshape([regions.centroid], 2, numel(regions))';
region_ids = [areas centroids];
[region_ids unique_ids] = unique(region_ids, 'rows');
regions = regions(unique_ids);

% find subregions of each region  
nregions = numel(regions);
existing_subregion_centroids = [];
for i = 1 : nregions 
    regions(i).regionID = i;
    % find all subregions in the region i.
    [regions(i).subregions] = getSubRegions(regions(i), size(img), existing_subregion_centroids);
    
    % pick subregions which define the region's shape (i.e. contour dominant
    % subregions)
    regions(i).subregions = getShapeDefineSubRegions(regions(i));
    existing_subregion_centroids =...
        [existing_subregion_centroids; reshape([regions(i).subregions.centroid], 2, numel(regions(i).subregions))'];
end

% merge subregions from different regions which have the same contours in
% them.
subregions = [regions.subregions];
centroids = reshape([subregions.centroid], 2, numel(subregions))';
ncontours = [subregions.ncontours]';
subregion_props = [centroids ncontours];
[unique_subregion_props, uid_i, uid_j] = unique(subregion_props, 'rows');

n_unique_subregions = 1;
for i = 1 : numel(uid_j)
    same_subregion_inds = find(uid_j == uid_j(i) & uid_j(i) ~= -1);
    if ( ~isempty(same_subregion_inds) )
        same_subregion_ncontours = unique_subregion_props(uid_j(i), 3);
        subregion_contours =...
            reshape([subregions(same_subregion_inds).contours], same_subregion_ncontours, numel(same_subregion_inds))';
        [unique_subregion_contours uid_ii uid_jj] =...
                    unique(subregion_contours, 'rows', 'first');
        for j = 1 : numel(uid_jj)
            unique_subregion_inds = find(uid_jj == uid_jj(j) & uid_jj(j) ~= -1);
            if ( ~isempty(unique_subregion_inds) )
                unique_subregions(n_unique_subregions).parentIDs = [subregions(same_subregion_inds(unique_subregion_inds)).parentID];
                unique_subregions(n_unique_subregions).contours = subregions(same_subregion_inds(unique_subregion_inds(1))).contours;
                unique_subregions(n_unique_subregions).ncontours = subregions(same_subregion_inds(unique_subregion_inds(1))).ncontours;
                unique_subregions(n_unique_subregions).centroid = subregions(same_subregion_inds(unique_subregion_inds(1))).centroid;
                n_unique_subregions = n_unique_subregions + 1;
            end
            uid_jj(unique_subregion_inds) = -1;
        end
        % invalidate subregions considered at this time
        uid_j(same_subregion_inds) = -1;
    end
end

%sub-sample contour map 
contour_inds = find(contour_map == 1);
ncontours = numel(contour_inds);
sample_inds = randperm(ncontours);
nsamples = min([max_contour_samples, floor(sample_rate*ncontours)]);
contour_map(contour_inds(sample_inds(nsamples+1:end))) = 0;
% adjust contour pixels in each unique_subregions
for i = 1 : numel(unique_subregions)
    subregion_contour_map = false(size(contour_map));
    subregion_contour_map(unique_subregions(i).contours) = 1;
    subregion_contour_map = contour_map & subregion_contour_map;
    unique_subregions(i).contours = find(subregion_contour_map == 1);
    unique_subregions(i).ncontours = numel(unique_subregions(i).contours);
end
% adjust contour pixels in each region
for i = 1 : numel(regions)
    region_contour_map = false(size(contour_map));
    region_contour_map(regions(i).contour) = 1;
    region_contour_map = contour_map & region_contour_map;
    regions(i).sampled_contour = find(region_contour_map==1);
    regions(i).nsampled_contour = numel(regions(i).sampled_contour);
end
% indexing contour pixels of each region and subregion into the contour map
contour_inds = find(contour_map == 1);
for i = 1 : numel(regions)
    [tf regioncontour2imagecontour_inds] = ismember(regions(i).sampled_contour, contour_inds);
    regions(i).sampled_contours_mapping_inds = regioncontour2imagecontour_inds;
end
for i = 1 : numel(unique_subregions)
    [tf regioncontour2imagecontour_inds] = ismember(unique_subregions(i).contours, contour_inds);
    unique_subregions(i).contours_mapping_inds = regioncontour2imagecontour_inds;
end
    
end % function end

function subregions = getSubRegions(region, imgsize, existing_centroids)
% sample 64 by 64 sub-regions of the input region
sample_rate = 8;

%     x_min = floor(region.boundingbox(1));
%     y_min = floor(region.boundingbox(2));
%     x_min = x_min - mod(x_min,sample_rate);
%     y_min = y_min - mod(y_min,sample_rate);
%     w = ceil(region.boundingbox(3));
%     h = ceil(region.boundingbox(4));
%     x_max = x_min + w;
%     y_max = y_min + h;
%     x_max = x_max + (sample_rate - mod(x_max,sample_rate));
%     y_max = y_max + (sample_rate - mod(y_max,sample_rate));
%   
%     [interior_y, interior_x] = ind2sub(imgsize, region.interior);

    [contour_y, contour_x] = ind2sub(imgsize, region.contour);
    candidate_centroids_inds = find(mod(contour_x, sample_rate) == 0 | mod(contour_y, sample_rate) == 0);
    [candidate_centroids_y, candidate_centroids_x] = ind2sub(imgsize, region.contour(candidate_centroids_inds));
    common_centroids = intersect(existing_centroids, [candidate_centroids_x candidate_centroids_y], 'rows');
    
    subregion_count = 1;
    for i = 1 : size(common_centroids, 1)
        x = common_centroids(i,1);
        y = common_centroids(i,2);
        if ( x ~= -1 && y ~= -1)
            contour_inds = x-32 <= contour_x & contour_x <= x+32 & y-32 <= contour_y & contour_y <= y+32;
            included_candidate_inds = x-32 <= candidate_centroids_x &...
                                     candidate_centroids_x <= x+32 &...
                                     y-32 <= candidate_centroids_y &...
                                     candidate_centroids_y <= y+32;
            included_common_centroid_inds = x-32 <= common_centroids(:,1) &...
                                     common_centroids(:,1) <= x+32 &...
                                     y-32 <= common_centroids(:,2) &...
                                     common_centroids(:,2) <= y+32;
            ncontours = sum(contour_inds);
            contours =   region.contour(contour_inds);
            subregions(subregion_count).centroid = [x, y];
            subregions(subregion_count).ncontours = ncontours;
            subregions(subregion_count).contours = contours;
            subregions(subregion_count).parentID = region.regionID;
            candidate_centroids_inds(included_candidate_inds) = -1;
            common_centroids(included_common_centroid_inds,:) = -1;
            subregion_count = subregion_count + 1;
        end
    end
        
    for i = 1 : numel(candidate_centroids_inds)
        if ( candidate_centroids_inds(i) ~= -1 )
            x = candidate_centroids_x(i);
            y = candidate_centroids_y(i);
            contour_inds = x-32 <= contour_x & contour_x <= x+32 & y-32 <= contour_y & contour_y <= y+32;
            included_candidate_inds = x-32 <= candidate_centroids_x &...
                                     candidate_centroids_x <= x+32 &...
                                     y-32 <= candidate_centroids_y &...
                                     candidate_centroids_y <= y+32;
            ncontours = sum(contour_inds);
            contours =   region.contour(contour_inds);
            subregions(subregion_count).centroid = [x, y];
            subregions(subregion_count).ncontours = ncontours;
            subregions(subregion_count).contours = contours;
            subregions(subregion_count).parentID = region.regionID;
            candidate_centroids_inds(included_candidate_inds) = -1;
            subregion_count = subregion_count + 1;
        end
    end
%     for x = x_min : sample_rate : x_max
%         for y = y_min : sample_rate : y_max
%             %ninteriors = sum(x-32 <= interior_x & interior_x <= x+32 & y-32 <= interior_y & interior_y <= y+32);
%             contour_inds = x-32 <= contour_x & contour_x <= x+32 & y-32 <= contour_y & contour_y <= y+32;
%             ncontours = sum(contour_inds);
%             contours =   region.contour(contour_inds);
%             subregions(subregion_count).centroid = [x, y];
%             %subregions(subregion_count).ninteriors = ninteriors;
%             subregions(subregion_count).ncontours = ncontours;
%             subregions(subregion_count).contours = contours;
%             subregions(subregion_count).parentID = region.regionID;
%             subregion_count = subregion_count + 1;
%         end
%     end
end % end function

function shape_define_subregions = getShapeDefineSubRegions(region)
    ncontours = [region.subregions.ncontours];
    
    %identify the subregions containing the same number of contour pixels
    [unique_ncontours, unique_ids_i, unique_ids_j] = unique(ncontours);
    n_unique_subregions = 1;
    for i = 1 : numel(unique_ids_j)
        same_subregion_inds = find(unique_ids_j == unique_ids_j(i) & unique_ids_j(i) ~= -1);
        if ( ~isempty(same_subregion_inds) )
            % number of contour pixels --> a single number for all
            % candidate subregions
            same_subregion_ncontours = unique_ncontours(unique_ids_j(i));
            % indices of contour pixels of each candidate subregions
            subregion_contours = reshape([region.subregions(same_subregion_inds).contours], same_subregion_ncontours, numel(same_subregion_inds))';
            % find unique subregions which have different contour pixels
            [unique_subregion_contours uid_i] =...
                unique(subregion_contours, 'rows', 'first');
            % record indices of unique subregions 
            uids(n_unique_subregions : n_unique_subregions + numel(uid_i) - 1) =...
                same_subregion_inds(uid_i);
            n_unique_subregions = n_unique_subregions + numel(uid_i);
            % invalidate subregions considered at this time
            unique_ids_j(same_subregion_inds) = -1;
        end
    end
    
    shape_define_subregions =  region.subregions(uids);
    ncontours = [shape_define_subregions.ncontours];
    shape_define_subregions = shape_define_subregions(ncontours > 64*0.9); % select subregions that have contour pixels more than 64*3.
                                                                                                               % 64 means that parent region size is 64 by 64.
end % end function 
