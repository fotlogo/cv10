function [regions unique_subregions] = test(regions)

%region_centers = reshape([regions.centroid], nregions,2);
%region_area = [regions.area]';
region_perims = [regions.perimeter];
valid_regions = region_perims < 24*4;
regions = regions(valid_regions);
nregions = numel(regions);

%region_centers = region_centers(valid_regions,:);
%region_area = region_centers(valid_regions,:);
%region_props = [region_centers region_area];
%[unique_region_props, unique_region_ids] = unique(region_props, 'rows');
%unique_regions = regions(unique_region_ids);

n = 1;
for i = 1 : nregions
    nsubs = numel(regions(i).subregions);
    if (nsubs > 0 ) 
        temp(n:n+nsubs-1, :) = reshape([regions(i).subregions.centroid], nsubs, 2);
        n = n+nsubs;
    end
end
unique_subregions = unique(temp, 'rows');