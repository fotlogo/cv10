% test of bounding box

fname = 'data/attribute_data/ayahoo_test.txt';
img_dir = 'data/ayahoo_test_images';
[img_names img_classes bboxes attributes] = read_att_data(fname);

start=150;
count=50;

for i=1+start:count+start
    points_hor=bboxes(i,1):1:bboxes(i,3);
    points_ver=bboxes(i,2):1:bboxes(i,4);
    img=imread([img_dir,'/',img_names{i}]);
    for j=1:length(points_ver)
        img(points_ver(j),bboxes(i,1),:)=uint8([0,255,0]);
        img(points_ver(j),bboxes(i,3),:)=uint8([0,255,0]);
    end
    for j=1:length(points_hor)
        img(bboxes(i,2),points_hor(j),:)=uint8([0,255,0]);
        img(bboxes(i,4),points_hor(j),:)=uint8([0,255,0]);    
    end
    figure,imshow(img)
end