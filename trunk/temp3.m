% temp.m
% calculate precision

% path
BasePath='/u/atian/cv/final/code/';
image_set='apascal';
img_dir = [BasePath,'data/',image_set,'_images'];
result_dir=[BasePath,'out/seg_out/'];

% read bounding boxes of some class
load('data_exp')
cate=1; % car

% get the dir of results
result_files=dir(result_dir);
result_files=char(result_files.name);
result_files=result_files(4:end,:);
result_idx=[];
for i=1:length(result_files)
    result_idx(i,1)=str2num(result_files(i,6:11));
end

% parameter
par='_6_1.5_';

recall_obj=[];
precision_seg=[];

for i=1:size(data_exp.names,1)
    if (data_exp.classes{i}.class(cate)==0)
        continue;
    end
    
    img_name=data_exp.names{i};
    img_name=regexprep(img_name, '\.jpg', '');
    
    img_idx=str2num(img_name(6:11));
    
    isExist=(result_idx==img_idx);
    if sum(isExist)==0
        continue;
    end
    
    % get image related dir
    result_list_dir=result_files(isExist,:);
    result_data_dir=[];
    for j=1:size(result_list_dir,1)
        if strncmp(result_list_dir(j,:),[img_name,par],16)>0
            result_data_dir=result_list_dir(j,:);
            break;
        end
    end
    if sum(size(result_data_dir))==0
        continue;
    end
    
    result_data_dir=regexprep(result_data_dir, ' ', '');
    if exist([result_dir,result_data_dir,'/obj_seg.mat'],'file')<=0
        continue;
    end
    load([result_dir,result_data_dir,'/obj_seg'])
    % if object is not detected
    if sum(size(obj_seg.mask))==0
        recall_obj=[recall_obj,0];
    else
        recall_obj=[recall_obj,1];
        
        % START FROM HERE
        % combine bounding boxes
        groundTruth=zeros(size(obj_seg.mask));
        bboxes=data_exp.bboxes{i}.box{cate};
        for j=1:length(bboxes.coor)
            temp=bboxes.coor{j};
            groundTruth(temp(2):temp(4),temp(1):temp(3))=1;
        end
        
        % get the segmentation precison
        interSeg=sum(sum(obj_seg.mask.*groundTruth));
        temp_seg=obj_seg.mask+groundTruth;
        temp_seg=temp_seg>0;
        unionSeg=sum(sum(temp_seg));
        precision_seg=[precision_seg,interSeg/unionSeg];
    end    
end
disp('Total images')
disp(length(recall_obj))
disp('recall object')
disp(mean(recall_obj))
disp('precision segmentation')
disp(mean(precision_seg))    

