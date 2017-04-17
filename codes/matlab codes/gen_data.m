%% Generate patches for 3D data
%% This script divides a 3D volume into three categories
%% 1. 2D patches around every pixel in a slice by slice manner
%% 2. 3D tri patches
%% 3. Full 3D patches
clear all
clc
%% Loading big data
clear all
clc
i=0;
cbct=[];
label=[];
for n=1:4; % File Name
filename1=sprintf('reconstruction%d.mat',n);
filename2=sprintf('materials_%d.mat',n);
x=load(filename2); %    Load labels
materials=x.materials(:,:,37:end); 

y=load(filename1);
reconstruction=y.reconstruction;
reconstruction=equalizeVoxel(reconstruction); %% equalize voxels
reconstruction=reconstruction(:,:,37:end);
cbct(:,:,i+1:i+30)=reconstruction(:,:,11:40); % Take only first 70 slices. They are important
label(:,:,i+1:i+30)=materials(:,:,11:40);
i=i+30;
n
end
cbct=double(cbct);
clearvars x y reconstruction filename1 filename2 i materials
%%
%% Load single Data
n=1; % File Name
 filename1=sprintf('reconstruction%d.mat',n);
filename2=sprintf('Seg_CT_%d.mat',n);
x=load(filename2); %    Load labels
materials=x.CT(:,:,37:end); 

y=load(filename1);
reconstruction=y.reconstruction;
reconstruction=equalizeVoxel(reconstruction); %% equalize voxels
reconstruction=reconstruction(:,:,37:end);
cbct=reconstruction(:,:,1:50); % Take only first 70 slices. They are important
materials=materials(:,:,1:50);
clearvars x y reconstruction filename1 filename2 i
%% Padding
n=33; % patch dimensions. it is cube - must be odd
m=floor(n/2); %% used for padding
cbct=padarray(cbct,[0 0 m]);
% cbct=cbct(:,:,1:end-m);
materials=padarray(label,[0 0 m]); %only padd along the depth
% materials=materials(:,:,1:end-m);
materials=changem(materials,[1 2 3],[0 1 2]);
%% 
cbct_copy=cbct;
  %%% Make image binary
cbct(cbct<=0.5)=false;
cbct(cbct>0.5)=true;
se = strel('disk',10);
% se = strel('disk',10);
cbct2 = imclose((cbct),se);
cbct3=imfill(cbct2,'holes');
% imshow3D(cbct2)
% ct(32,:,:)=0;
CC = bwconncomp(cbct3);
stats=regionprops(CC,'PixelIdxList');
for i=1:size(stats,1)
    x=numel(stats(i).PixelIdxList);
    y(i)=x;
end
    [score index]=max(y); %%% find biggest area
    
idx=stats(index).PixelIdxList;
new_cbct=zeros(512,512,size(cbct,3));
new_cbct(idx)=cbct_copy(idx);

clearvars -except new_cbct materials label idx m


%% bringing it under 0-1 range
cbct=mat2gray(new_cbct,[0 2.5]);
clearvars new_cbct


%% Find edges
edge=canny(cbct,'sigma',0.5,'thresh',0.2);
[a b c]=size(edge);
edge(:,:,m:m+1)=0;
edge(:,:,c-m:c-m+1)=0;
figure(1)
imshow3D(edge)
edgeidx=find(edge==1); %Find edge indixes


%% Taking only the edge samples for training 
data=cbct(edgeidx);
labels=materials(edgeidx);


 %% remove the border edges caused due to padding
% [a b c]=ind2sub(size(cbct),edgeidx);
% Edge_coord=horzcat(a,b,c);
% edgeidx(edgeidx<4300000)=0; % incase you need to remove the first slice edge pixels only
%  edgeidx(edgeidx==0)=[];
 
 %% Find bone tissue air idx from the idx array
%  materials=changem(materials,[1 2 3],[0 1 2]);
 x=materials(edgeidx);
%  idxLabels=horzcat(idx,x);
A = find(x==1);
airIndex=edgeidx(A);
T= find(x==2);
tissueIndex=edgeidx(T);
B = find(x==3);
boneIndex=edgeidx(B);

indx=size(airIndex,1);


permute=randperm(size(airIndex,1));
airIndex=airIndex(permute,:);
airidx=airIndex(1:indx,:);
permute=randperm(size(tissueIndex,1));
tissueIndex=tissueIndex(permute,:);
tissueidx=tissueIndex(1:indx,:);
permute=randperm(size(boneIndex,1));
boneIndex=boneIndex(permute,:);
boneidx=boneIndex(1:indx,:);

data_indexes=vertcat(boneidx,tissueidx,airidx);
Data=unique(data_indexes,'rows');
clearvars -except Data materials cbct
    %% Generating training data and labels
data=cbct(Data);
labels=materials(Data);
% labels=changem(labels,[1 1 2],[1 2 3]); %% Making it binary
figure(3)
histogram(data)
figure(4)
histogram(labels)

%% See this
x=zeros(size(cbct));
x(Data)=cbct(Data);
y=zeros(size(cbct));
y(Data)=materials(Data);
figure(5)
imshow3D(x)
figure(6)
imshow3D(y)

%% Get the training data as per defined by the input
%% Get TRi patches for the Data
sz=16; % Determines size of patch. THIS IS THE LENGTH OF THE PATCH above the pixel of interest
[a b c]=ind2sub(size(cbct),Data);
Data_coord=horzcat(a,b,c);
clearvars -except cbct Data Data_coord materials labels sz
j=1;

tic
for i=1:size(Data,1)
[out ind]=gettripatch(cbct,Data_coord(i,:),sz,Data(i));
if size(out,1)==1
    continue
else
    patch(:,:,:,j)=out;
index(j)=ind;
    j=j+1;
end
end
toc

%% Get full 3D patch around the Data
sz=12; % Determines size of patch. THIS IS THE LENGTH OF THE PATCH above the pixel of interest
[a b c]=ind2sub(size(cbct),Data);
Data_coord=horzcat(a,b,c);
clearvars -except cbct Data Data_coord materials labels sz
j=1;

tic
for i=1:size(Data,1)
[out ind]=get3dpatch(cbct,Data_coord(i,:),sz,Data(i));
if size(out,1)==1
    continue
else
    patch(:,:,:,j)=out;
index(j)=ind;
    j=j+1;
end
end
toc

%% Get 2D patch around the Data
sz=16; % Determines size of patch. THIS IS THE LENGTH OF THE PATCH above the pixel of interest
[a b c]=ind2sub(size(cbct),Data);
Data_coord=horzcat(a,b,c);
clearvars -except cbct Data Data_coord materials labels sz
j=1;

tic
for i=1:size(Data,1)
[out ind]=get2dpatch(cbct,Data_coord(i,:),sz,Data(i));
if size(out,1)==1
    continue
else
    patch(:,:,j)=out;
index(j)=ind;
    j=j+1;
end
end
toc

%% permuting
label=materials(index);
% label=changem(label,[1 2 3],[0 1 2]);
% permute=randperm(numel(index));
% data=patch(:,:,:,permute);
% label=label(:,permute);

%% so we put them in one struct imdb

% x=find(label==3);
% y=find(label==2);
% z=find(label==1);
% 
% bone=data(:,:,:,x);
% tissue=data(:,:,:,y);
% air=data(:,:,:,z);
% v=ones(1,numel(x));
% train_data=cat(4,bone,tissue(:,:,:,1:numel(x)),air(:,:,:,1:numel(x)));
% train_labels=single(horzcat((v.*3),(v.*2),v));


%% Permute again
% permute=randperm(size(train_labels,2));
% 
% imdb.images.data=train_data(:,:,:,permute);
% imdb.images.labels=train_labels(:,permute);
imdb.images.data=single(patch);
imdb.images.labels=single(labels)';
x=size(patch,4);
%%% generating ids
for i=1:x
    id(i)=i;
end
%%% generating sets
set=ones(x,1);
p=randperm(x);
pnew=p(1:100000);
set(pnew)=2;
histogram(set)
%%%%%%%%%%%
imdb.images.set=set';
imdb.images.id=id;
clearvars -except imdb
