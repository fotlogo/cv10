% function of histogram intersection
% vecBase is the pattern vector, vecTest is the test vector
% do not change the order


function sim=histIntersect(vecBase, vecTest)

temp=vecBase-vecTest;
sim=1-sum(temp(temp>0))/sum(vecBase);
