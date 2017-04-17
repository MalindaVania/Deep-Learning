function pred =CNN_2D_TEST(imdb);
%% TEST THE 3D cnn using triplanar images

%% Load test set
data=imdb.images.data;
batchSize=1000;

%% Load network
 net = load('data/CNN_2D/phantom_net.mat') ;

 
 %% Apply net to data in a batch wise manner
 [a b c]=size(imdb.images.data);
j=0;
pred=zeros(a,b);
 for t=1:batchSize:d
%     get next image batch and labels
        batch = imdb.images.id(t:min(t+batchSize-1, c)) ;
    im = imdb.images.data(:,:,batch) ;
im = reshape(im, a, b,1, []) ;
    res = vl_simplenn(net, (gpuArray(im))) ;
    out=gather(res(end).x);

[score,pr] = max(out,[],3);
pred(1,j+1:j+size(pr,5))=squeeze(pr)';
j=j+size(pr,5);
   
 end
%  prediction=reshape(pred,[1,numel(pred)]);
 error=(nnz(imdb.images.labels-pred)/numel(pred))*100
%  %% Generate Batch Function
%  function [im, label] = getBatch(imdb, batch)
% % --------------------------------------------------------------------
% [a b c d]=size(imdb.images.data);
% im = imdb.images.data(:,:,:,batch) ;
% im = reshape(im, a, b, c,1, []) ;
% label = imdb.images.labels(:,batch) ;