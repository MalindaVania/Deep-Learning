clear all
clc

i=load('data/phantom6.mat');

im=i.images.data;
im=im(:,:,1736);
l=i.images.label;
label=l(:,:,1736);
im=im(1:32,1:32);
figure(1)
subplot(1,2,1)
imshow(im,[]);
subplot(1,2,2)
imshow(label,[])

f1=gpuArray(single(randn(7,7,7,1,20)));
a=gpuArray(single(randn(24,24,24)));
B=gpuArray(single(ones(1,20)));
y=mex_conv3d(a,f1,B);
pool   = [2,2,2];          % 3D pooling window size
stride = [2,1,2];
[h idx]= mex_maxpool3d(y,...
  'pool',pool);

out_conv_1=vl_nnconv(im,f1,[],'stride',1);
vl_imarraysc(squeeze(out_conv_1),'spacing',2)
axis equal ; title('filters in the first layer') ;
 out_deconv_1=vl_nnconvt(out_pool_2,f1,[],'upsample',2);
% imshow(out_deconv_1,[]);
figure(2)
subplot(1,2,1)
imshow3D(out_conv_1);
figure(3)
subplot(1,2,1)
imshow(im,[]);
subplot(1,2,2)
imshow(out_deconv_1,[])


pool1=[2 2];
out_pool_2=vl_nnpool(out_conv_1,pool1,'stride',2);

f2=single(randn(5,5,20,50));
out_conv_3=vl_nnconv(out_pool_2,f2,[],'stride',1);

pool2=[2 2];
out_pool_4=vl_nnpool(out_conv_3,pool2,'stride',2);

f3=single(randn(4,4,50,500));
out_conv_5=vl_nnconv(out_pool_4,f3,[],'stride',1);

out_relu_6=vl_nnrelu(out_conv_5);

f4=single(randn(2,2,500,3));
out_conv_7=vl_nnconv(out_relu_6,f4,[],'stride',1);

c=[1 2 3];
c=single(ones(33));
out_softmaxloss_8=vl_nnsoftmaxloss(out_conv_7,1);