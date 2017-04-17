function pred =CNN_3D_TEST(imdb);
%% TEST THE 3D cnn using triplanar images

%% Load test set
data=imdb.images.data;
batchSize=500;

%% Load network
 net = load('data/CNN_3D/phantom_net.mat') ;

 
 %% Apply net to data in a batch wise manner
 [a b c d]=size(imdb.images.data);
j=0;
pred=zeros(1,d);
 for t=1:batchSize:d
%     get next image batch and labels
        batch = imdb.images.id(t:min(t+batchSize-1, d)) ;
    im = imdb.images.data(:,:,:,batch) ;
im = reshape(im, a, b, c,1, []) ;
    res = vl_simplenn(net, (gpuArray(im))) ;
    out=gather(res(end).x);

[score,pr] = max(out,[],4);
pred(1,j+1:j+size(pr,5))=squeeze(pr)';
j=j+size(pr,5);
   
 end
%  prediction=reshape(pred,[1,numel(pred)]);
 error=(nnz(imdb.images.labels-pred)/numel(pred))*100
 %% Generate Batch Function
 function [im, label] = getBatch(imdb, batch)
% --------------------------------------------------------------------
[a b c d]=size(imdb.images.data);
im = imdb.images.data(:,:,:,batch) ;
im = reshape(im, a, b, c,1, []) ;
label = imdb.images.labels(:,batch) ;