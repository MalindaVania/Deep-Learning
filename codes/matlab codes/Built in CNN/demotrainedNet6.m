%% This code refers to the data preparation, training and testing of the trainedNet6.mat stored in the Results folder.

%% Loading big data
clear all
clc
i=0;
cbct=[];
label=[];
for n=1:20; % File Name
filename1=sprintf('reconstruction%d.mat',n);
filename2=sprintf('materials_%d.mat',n);
x=load(filename2); %    Load labels
materials=x.materials(:,:,37:end); 

y=load(filename1);
reconstruction=y.reconstruction;
reconstruction=equalizeVoxel(reconstruction); %% equalize voxels
reconstruction=reconstruction(:,:,37:end);
cbct(:,:,i+1:i+150)=reconstruction(:,:,1:150); % Take only first 70 slices. They are important
label(:,:,i+1:i+150)=materials(:,:,1:150);
i=i+150;
n
end
cbct=double(cbct);
clearvars x y reconstruction filename1 filename2 i materials n


%% im2col using for neighbors
%% This loop was not completed. I had to stop at i=1912645
n=33; % patch dimensions. it is square - must be odd
m=floor(n/2);
out=[];
targets=[];
for i=1:7:size(cbct,3)
 B=padarray(cbct(:,:,i),[m m]);
C=im2col(B,[n n],'sliding'); %%% generate the neghbours for every pixel in a column matrix
% Find Edges
edge=canny(cbct(:,:,i),'sigma',0.5,'thresh',0.2);
edgeidx=find(edge==1); %Find edge indixes
Y=label(:,:,i);
Z=Y(edgeidx);
x=C(:,edgeidx);
out=horzcat(out,x);
targets=vertcat(targets,Z);
clearvars B C Y Z x edge edgeidx
i
end
targets=changem(targets,[1 2 3],[0 1 2]);
[rows columns]=size(out);
 data=reshape(out,[n n 1 columns]);

clearvars -except targets data n m
 
%% Training
layers = [imageInputLayer([33 33 1],'Normalization','none');
          convolution2dLayer(5,20);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          convolution2dLayer(5,32);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(512);
          reluLayer();
          fullyConnectedLayer(3);
          softmaxLayer();
          classificationLayer()];
  opts = trainingOptions('sgdm', ...
          'LearnRateSchedule', 'piecewise', ...
           'LearnRateDropFactor', 0.5, ... 
          'LearnRateDropPeriod', 5, ... 
           'MaxEpochs',10, ... 
          'MiniBatchSize', 500);
      [a b c]=size(data);
      data=reshape(data,[a b 1 c]);
      targets=nominal(targets);
      tic;
      [trainedNet6,trainedInfo] = trainNetwork(data,targets,layers,opts)
      toc;
      clearvars -except trainedNet5 trainedInfo data targets
%% test
n=21;
filename1=sprintf('reconstruction%d.mat',n);
filename2=sprintf('materials_%d.mat',n);
x=load(filename2); %    Load labels
materials=x.materials(:,:,37:end); 

y=load(filename1);
reconstruction=y.reconstruction;
reconstruction=equalizeVoxel(reconstruction); %% equalize voxels
reconstruction=reconstruction(:,:,37:end);
test=reconstruction;
clearvars reconstruction y x filename1 filename2

k=1;
n=33; % patch dimensions. it is square - must be odd
m=floor(n/2);
tic;
for i=1:150
    
 B=padarray(test(:,:,i),[m m]);
C=im2col(B,[n n],'sliding'); %%% generate the neghbours for every pixel in a column matrix
[rows columns]=size(C);

for j=1:columns
    Tdata(:,:,j)=reshape(C(:,j),[n n]);
end
            testset=reshape(Tdata,[n n 1 size(Tdata,3)]);
     
            YTest = classify(trainedNet5,testset,'MiniBatchSize',1000);
    
            prediction=single(YTest);
      predicted_volume(:,:,k)=reshape(prediction,[512 512]);
k=k+1;
i
clearvars B C prediction YTest Tdata j 

end
toc;

pv=changem(predicted_volume,[0 1 2],[1 2 3]);
figure(1)
imshow3D(pv)
figure(2)
imshow3D(materials)
diff=nnz(pv-materials(:,:,1:150))/numel(pv); % Total Error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% K sample T test
[h p]=kstest2(mat2gray(reshape(pv,[1 numel(pv)]))',mat2gray(reshape(test_materials,[1 numel(pv)]))','Alpha',0.001)
