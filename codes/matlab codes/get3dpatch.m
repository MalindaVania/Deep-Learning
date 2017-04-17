function [neighbors ind] = get3dpatch(imdb,coord,k,index)
 siz = size(imdb);                              %# matrix size
%# 3D point location

%# neighboring points
%# radius size
dim=k*2 +1;
[sx,sy,sz] = ndgrid(-k:k,-k:k,-k:k);          %# steps to get to neighbors
xyz = bsxfun(@plus, coord, [sx(:) sy(:) sz(:)]);  %# add shift
xyz = bsxfun(@min, max(xyz,1), siz);          %# clamp coordinates within range
xyz = unique(xyz,'rows');                     %# remove duplicates
%  xyz(ismember(xyz,p,'rows'),:) = [];           %# remove point itself

%# show solution
% figure
% line(p(1), p(2), p(3), 'Color','r', ...
%     'LineStyle','none', 'Marker','.', 'MarkerSize',50)
% line(xyz(:,1), xyz(:,2), xyz(:,3), 'Color','b', ...
%     'LineStyle','none', 'Marker','.', 'MarkerSize',20)
% view(3), grid on, box on, axis equal
% axis([1 siz(1) 1 siz(2) 1 siz(3)])
% xlabel x, ylabel y, zlabel z

linearInd = sub2ind(siz, xyz(:,1), xyz(:,2), xyz(:,3));
if numel(linearInd)==(dim*dim*dim)
neighbors=reshape(imdb(linearInd),[dim dim dim]);
ind=index;
else
    neighbors=1;
    ind=0;
end
