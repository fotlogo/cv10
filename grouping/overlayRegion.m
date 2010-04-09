function overlayRegion(img, regions, usubregions)

close all;

areas = [regions.area];
[areas area_rank] = sort(areas, 'descend'); 
pids = {usubregions.parentIDs};

for i = 1 : numel(areas)

    region_idx = area_rank(i);
    cell_region_idx = num2cell(repmat(region_idx, 1, numel(pids)));
    cell_subregion_inds = cellfun(@(x,y) sum((x==y)) > 0, pids, cell_region_idx, 'UniformOutput',false); 
    subregion_inds = cell2mat(cell_subregion_inds);
    interior = img;
    contour = img;

    imR = interior(:,:,1);
    imG = interior(:,:,2);
    imB = interior(:,:,3);
    imR(regions(region_idx).interior) = 255;
    imG(regions(region_idx).interior) = 0;
    imB(regions(region_idx).interior) = 0;

    interior(:,:,1) = imR;
    interior(:,:,2) = imG;
    interior(:,:,3) = imB;

    subplot(1,2,1); imshow(interior);

    imR = contour(:,:,1);
    imG = contour(:,:,2);
    imB = contour(:,:,3);
    imR(regions(region_idx).contour) = 255;
    imG(regions(region_idx).contour) = 0;
    imB(regions(region_idx).contour) = 0;

    contour(:,:,1) = imR;
    contour(:,:,2) = imG;
    contour(:,:,3) = imB;
    
    hold off;
    subplot(1,2,2); imshow(contour);
    nsubregions = sum(subregion_inds);
    centroids = reshape([usubregions(subregion_inds).centroid], 2, nsubregions)';
    parentIDs  = pids(subregion_inds);
    if (~isempty(centroids) )
        hold on;
        m = ceil(median(1:nsubregions));
        for j = 1 : nsubregions
            center = centroids(j, :);
            if ( numel(parentIDs{j}) ==1 )
                rectangle('Position', [center(1)-2, center(2)-2, 5, 5], 'LineWidth', 1, 'FaceColor', [0,1,1]);
            else
                rectangle('Position', [center(1)-2, center(2)-2, 5, 5], 'LineWidth', 1, 'FaceColor', [1,1,0]);
            end
                
            if ( j == m ) 
                rectangle('Position', [center(1)-32, center(2)-32, 64, 64], 'LineWidth', 2, 'EdgeColor', [1,0,1]);
            end
        end
        
            
%         [shape_regions shape_regions_idx] = sort([regions(region_idx).subregions.ncontours], 'descend');
%         nshape_regions = sum(shape_regions > 64*5); % select subregions that have contour pixels more than 64*5.
%         max_interiors = max([regions(region_idx).subregions.ninteriors]);
%         hold on;
%         if ( max_interiors < 64*64*0.25 ) 
%             for j = 1 : nshape_regions
%                 % 3/4(0.75) threshold is not considered since this region
%                 % will almost consist of contour pixels in a small area;
%                 % subregion should contain as many contour pixels as
%                 % possible without removing overlapping subregions.
%                 if ( regions(region_idx).subregions(shape_regions_idx(j)).ninteriors > regions(region_idx).area*0.25)
%                     center = regions(region_idx).subregions(shape_regions_idx(j)).centroid;
%                     rectangle('Position', [center(1)-2, center(2)-2, 5, 5], 'LineWidth', 1, 'FaceColor', [0,1,1]);
%                     % for seeing how large 64 by 64 pixels are.
%                     if ( j == 1 )
%                         rectangle('Position', [center(1)-32, center(2)-32, 64, 64], 'LineWidth', 2, 'EdgeColor', [1,0,1]);
%                     end
%                 end
%             end
%         else
%             for j = 1 : nshape_regions
%                 % select subregions whose interior pixels are less than 3/4 and more than 1/4 of
%                 % the total area
%                 % upper and lower threshold aim to remove unnecessay
%                 % overlapping subregions which share the same contour
%                 % pixels.
%                 if ( regions(region_idx).subregions(shape_regions_idx(j)).ninteriors < 64*64*0.75 &&...
%                      regions(region_idx).subregions(shape_regions_idx(j)).ninteriors > 64*64*0.25)
%                     center = regions(region_idx).subregions(shape_regions_idx(j)).centroid;
%                     rectangle('Position', [center(1)-2, center(2)-2, 5, 5], 'LineWidth', 1, 'FaceColor', [0,1,1]);
%                     % for seeing how large 64 by 64 pixels are.
%                     if ( j == 1 )
%                         rectangle('Position', [center(1)-32, center(2)-32, 64, 64], 'LineWidth', 2, 'EdgeColor', [1,0,1]);
%                     end
%                 end
%             end
%         end
    else
        title('no subregion');
    end
    
    %disp(['area: ' num2str(regions(region_idx).area)]);
    %disp(['perimeter: ' num2str(regions(region_idx).perimeter)]);
    disp([num2str(i) 'th region out of  ' num2str(numel(areas)) ]);
    pause;
end
%disp(['# of valid regions: ' num2str(nvalid_regions)]);

pause;
close;

