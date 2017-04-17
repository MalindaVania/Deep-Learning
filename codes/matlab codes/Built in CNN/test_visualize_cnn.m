%% Test CNN
%% Load Test Set
f=1;
% files=[1 4 20 25 30];
for file=21:40
% for z=1:5
%     file=files(z);
filename1=sprintf('reconstruction%d.mat',file);
filename2=sprintf('materials_%d.mat',file);
x=load(filename2); %    Load labels
test_materials=x.materials(:,:,37:end); 
y=load(filename1);
reconstruction=y.reconstruction;
reconstruction=equalizeVoxel(reconstruction); %% equalize voxels
reconstruction=reconstruction(:,:,37:end);
test=reconstruction;
clearvars x y filename1 filename2 reconstruction
%% Gaussian Filtering
%test = imgaussfilt3((test),0.5,'FilterSize',3,'FilterDomain','spatial');
%% Perform Cleaning - Get rid of noisy pixels around the image
test_copy=test;
  %%% Make image binary
test(test<=0.5)=0;
test(test>0.5)=1;
% ct(32,:,:)=0;
CC = bwconncomp(test);
stats=regionprops(CC,'PixelIdxList');
for i=1:size(stats,1)
    x=numel(stats(i).PixelIdxList);
    y(i)=x;
end
    [score index]=max(y); %%% find biggest area
    
idx=stats(index).PixelIdxList;
new_cbct=zeros(512,512,size(test,3));
new_cbct(idx)=test_copy(idx);
test=new_cbct;
clearvars new_cbct idx CC stats test_copy x y score index i 
% tic
thresh=multithresh(test,2);
out=imquantize(test,thresh);
% sprintf('Auto Thresholding took %0.2f second for execution',t)
pv=changem(out,[0 1 2 2],[1 2 3 4]);
% t=toc;
% Start Test
n=33; % patch dimensions. it is square - must be odd
m=floor(n/2);
tic;
parfor i = 1:154
    B=padarray(test(:,:,i),[m m]);
    C=im2col(B,[33 33],'sliding');
    x=reshape(C,[33 33 1 262144]);
    YTest = classify(trainedNet6,x,'MiniBatchSize',3000); %
    prediction=single(YTest);
    predicted_volume(:,:,i)=reshape(prediction,[512 512]);
end
t=toc;
Visualization
  pv=changem(predicted_volume,[0 1 2],[1 2 3]);
  err(z,:)=ROI_error(pv,test_materials);
  clearvars -except files z err trainedNet6
% 
% figure(1)
% imshow3D(pv)
% figure(2)
% imshow3D(test_materials)

 diff=nnz(pv-test_materials)/numel(pv)*100; % Total Error
%%CP FOR Bone
caseB=changem(pv,[0 0 1],[0 1 2]);
caseB_L=changem(test_materials,[0 0 1],[0 1 2]);
CP_B=classperf(caseB_L,caseB,'Positive',1,'Negative',0);

%%CP FOR Tissue
caseT=changem(pv,[0 1 0],[0 1 2]);
caseT_L=changem(test_materials,[0 1 0],[0 1 2]);
CP_T=classperf(caseT_L,caseT,'Positive',1,'Negative',0);

%%CP FOR Air
caseA=changem(pv,[1 0 0],[0 1 2]);
caseA_L=changem(test_materials,[1 0 0],[0 1 2]);
CP_A=classperf(caseA_L,caseA,'Positive',1,'Negative',0);

TotalerrorCNN(1,f)=diff;

sensitivityThresh_B(1,f)=CP_B.Sensitivity*100;
specificityThresh_B(1,f)=CP_B.Specificity*100;
correct_rateThresh_B(1,f)=CP_B.CorrectRate*100;

sensitivityThresh_T(1,f)=CP_T.Sensitivity*100;
specificityThresh_T(1,f)=CP_T.Specificity*100;
correct_rateThresh_T(1,f)=CP_T.CorrectRate*100;

sensitivityThresh_A(1,f)=CP_A.Sensitivity*100;
specificityThresh_A(1,f)=CP_A.Specificity*100;
correct_rateThresh_A(1,f)=CP_A.CorrectRate*100;

% ex_timeThresh(1,f)=t;
f=f+1;
clearvars -except ex_timeThresh f trainedNet6 file TotalerrorThresh sensitivityThresh_B sensitivityThresh_T sensitivityThresh_A specificityThresh_B specificityThresh_T specificityThresh_A correct_rateThresh_B correct_rateThresh_T correct_rateThresh_A 
file
end

%% alogorithm 2: Adding bias field here now REPEAT THE SAME CONFUSION MATRIX
% tic
test = imgaussfilt3((test),0.5,'FilterSize',3,'FilterDomain','spatial');


% Perform 3D bias correction

% for k=2:2:50
x=test;
% y=find(x>1.2 & x<1.5);
% z=find(x>3);
x=mat2gray(x,[0 2]); 
x_copy=x;
 
% x_copy(y)=1.5;
% x_copy(z)=3;
% thresh=multithresh(x_copy,3);
% thresh(1)=0;
% warning('off','all');
tic
for i=1:154
%     thresh=multithresh(x_copy(:,:,i),3);
%     thresh(1)=0;
[bias(:,:,i) M(:,:,:,i)]=BCFCM2D(double(x_copy(:,:,i)),thresh,struct('p',2,'maxit',10,'epsilon',1e-5,'sigma',12,'alpha',1));
end
% for i=1:154
%     pv3(:,:,i)=max(M(:,:,:,i),[],3);
% end
% Bias Smoothing
% Sbias=imgaussfilt3(bias,2);
Corrected_cbct=x_copy-(bias);
thresh=multithresh(Corrected_cbct,2);
thresh(2)=thresh(2)-0.04;
seg=imquantize(Corrected_cbct,thresh);
t=toc;
  seg_FCM=changem(seg,[0 1 2],[1 2 3]);
 err(z,:)=ROI_error(pv,test_materials);
  clearvars -except files z err trainedNet6
end
% %sprintf('Bias Field correction and segmentation took %0.2f second for execution',t)
% pv=changem(seg,[0 1 2],[1 2 3]);
% %%%% END OF BIAS FIELD

diff=nnz(pv-test_materials)/numel(pv)*100; % Total Error
%%CP FOR Bone
caseB=changem(pv,[0 0 1],[0 1 2]);
caseB_L=changem(test_materials,[0 0 1],[0 1 2]);
CP_B=classperf(caseB_L,caseB);

%%CP FOR Tissue
caseT=changem(pv,[0 1 0],[0 1 2]);
caseT_L=changem(test_materials,[0 1 0],[0 1 2]);
CP_T=classperf(caseT_L,caseT);

%%CP FOR Air
caseA=changem(pv,[1 0 0],[0 1 2]);
caseA_L=changem(test_materials,[1 0 0],[0 1 2]);
CP_A=classperf(caseA_L,caseA);

% fprintf('TotalError=%f %% \n',diff)
% fprintf('Sensitivity=%f %% \n',CP.Sensitivity*100)
% fprintf('Specificity=%f %% \n',CP.Specificity*100)
% fprintf('CorrectRate=%f %% \n',CP.CorrectRate*100)
% fprintf('Time=%f s \n',t)
TotalerrorThresh(1,f)=diff;

sensitivityThresh_B(1,f)=CP_B.Sensitivity*100;
specificityThresh_B(1,f)=CP_B.Specificity*100;
correct_rateThresh_B(1,f)=CP_B.CorrectRate*100;

sensitivityThresh_T(1,f)=CP_T.Sensitivity*100;
specificityThresh_T(1,f)=CP_T.Specificity*100;
correct_rateThresh_T(1,f)=CP_T.CorrectRate*100;

sensitivityThresh_A(1,f)=CP_A.Sensitivity*100;
specificityThresh_A(1,f)=CP_A.Specificity*100;
correct_rateThresh_A(1,f)=CP_A.CorrectRate*100;

ex_timeThresh(1,f)=t;
f=f+1;
clearvars -except ex_timeThresh f trainedNet6 file TotalerrorThresh sensitivityThresh_B sensitivityThresh_T sensitivityThresh_A specificityThresh_B specificityThresh_T specificityThresh_A correct_rateThresh_B correct_rateThresh_T correct_rateThresh_A 
file
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(10)      
 vl_imarraysc(squeeze(trainedNet6.Layers(2,1).Weights),'spacing',1)
 axis equal ; title('First Layer Filters') ;set(gca,'xtick',[]);set(gca,'ytick',[])


%% Comparing with other algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Automatic Thresholding
% debug=0;
tic
thresh=multithresh(test,2);
out=imquantize(test,thresh);
% sprintf('Auto Thresholding took %0.2f second for execution',t)
pv=changem(out,[0 1 2],[1 2 3]);
t=toc;
% 
% if debug
% figure(3)
% imshow3D(out)
% figure(4)
% imshow3D(test)
% end
% diff=nnz(out-test_materials)/numel(pv)*100; % Total Error
% CP2=classperf(test_materials,out);
% fprintf('TotalError=%f %% \n',diff)
% fprintf('Sensitivity=%f %% \n',CP2.Sensitivity*100)
% fprintf('Specificity=%f %% \n',CP2.Specificity*100)
% fprintf('CorrectRate=%f %% \n',CP2.CorrectRate*100)
% fprintf('Time=%f s \n',t)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Kmeans
% warning('off','all')
% tic
% seg2=kmeans3D(test,3,1);
% t=toc;
% sprintf('K means took %0.2f second for execution',t)
% figure(4)
% imshow3D(seg2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bias field corrected Thresholding
tic
test = imgaussfilt3((test),0.5,'FilterSize',3,'FilterDomain','spatial');

test_copy=test;
test1=test;
  %%% Make image binary
test1(test1<=0.5)=0;
test1(test1>0.5)=1;
test=test1;
% ct(32,:,:)=0;
CC = bwconncomp(test);
stats=regionprops(CC,'PixelIdxList');
for i=1:size(stats,1)
    x=numel(stats(i).PixelIdxList);
    y(i)=x;
end
    [score index]=max(y); %%% find biggest area
    
idx=stats(index).PixelIdxList;
new_cbct=zeros(512,512,size(test,3));
new_cbct(idx)=test_copy(idx);
% Perform 3D bias correction

% for k=2:2:50
x=new_cbct;
% y=find(x>1.2 & x<1.5);
% z=find(x>3);
x=mat2gray(x,[0 2]); 
x_copy=x;
 
% x_copy(y)=1.5;
% x_copy(z)=3;
% thresh=multithresh(x_copy,3);
% thresh(1)=0;
warning('off','all');
tic
for i=1:154
    thresh=multithresh(x_copy(:,:,i),3);
    thresh(1)=0;
[bias(:,:,i) M]=BCFCM2D(double(x_copy(:,:,i)),thresh,struct('p',5,'maxit',10,'epsilon',1e-5,'sigma',3,'alpha',1));
end
toc
% Bias Smoothing
% Sbias=imgaussfilt3(bias,2);
Corrected_cbct=x_copy-(bias);
thresh=multithresh(Corrected_cbct,2);
thresh(2)=thresh(2)-0.04;
seg=imquantize(Corrected_cbct,thresh);
t=toc;
sprintf('Bias Field correction and segmentation took %0.2f second for execution',t)
seg=changem(seg,[0 1 2],[1 2 3]);
figure(5);
imshow3D(Corrected_cbct)
figure(6);
imshow3D(seg)
%% Comapring Bone pixels per slice for all except k-means
x = squeeze(sum(sum(pv==2, 1), 2));
y = squeeze(sum(sum(test_materials==2, 1), 2));
z = squeeze(sum(sum(out==2, 1), 2));
v = squeeze(sum(sum(seg==2, 1), 2));
% f = squeeze(sum(sum(pv2==2, 1), 2));
figure(9)
hold on
% plot(x,'b',z,'r',y,'g',v,'k')
% legend('Ref','CBCT','C_CBCT','CF_CBCT')
plot(x,'b')
plot(y, 'r')
plot(z, 'g')
plot(v, 'c')
% plot(f,'c')
% legend('CNN','Ref','Thres','FCM')
legend('Ref','Thres','FCM','CNN')
xlabel('Slice No.')
ylabel('No. of Bone Pixels')

% title('Visualizing number of bone pixels at every slice')

diff2=nnz(out-test_materials(:,:,1:154))/numel(pv); % Total Error
diff3=nnz(seg-test_materials(:,:,1:154))/numel(pv); % Total Error