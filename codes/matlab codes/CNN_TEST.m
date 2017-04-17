%%%% TESTING MY CNN
%%%% pATCH BASED LEARNING
clear all
clc

 net = load('data/patch_phantom_512/phantom_net.mat') ;

% I=load('data/phantom6');
% in=I.images.data;
% index=I.images.id;
% % im = im2single(imread('data/sentence-lato.png')) ;
% im = single(in) ;
%% Create test image
im = load('data/phantom.mat') ;
imdb=im.phantom;


images=imdb.images.data;
labels=imdb.images.label;
p=1;
i=2;
sz=512;

images=imdb.images.data;
labels=imdb.images.label;
p=1;
i=2;
    data_patch=gen_patches(images(:,:,i),sz);
    label_patch=gen_patches(labels(:,:,i),sz);
    for m=1:(512/sz)
        for n=1:(512/sz)
            data(:,:,p)=single(cell2mat(data_patch(n,m)));
            label(:,:,p)=single(cell2mat(label_patch(n,m)));
p=p+1;
        end
    end

    
       
clearvars -except data label net sz

%%
im=data;

test = 256*(im - net.imageMean) ;

opts.batchSize=512/sz;
opts.useGpu=true;
j=1;
for m=1:size(data,3)
    index(m)=m;
end

% Apply the CNN to the larger image
n=1;
 for t=1:opts.batchSize:numel(index)
%     get next image batch and labels
    batch = index(t:min(t+opts.batchSize-1, numel(index))) ;
    im = test(:,:,batch) ;
im =reshape(im, sz, sz, 1, []) ;
%     batch_time = tic ;
%     fprintf('training: epoch %02d: processing batch %3d of %3d ...', epoch, ...
%             fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;
%     [im, labels] = getBatch(i, batch) ;
%   net = vl_simplenn_move(net, 'cpu') ;

    res = vl_simplenn(net, (im)) ;
    
 %   [score(i),pred(x+n,y)] = max(squeeze(res(end).x(x,y,:,i))) ;
[score,pr] = max(res(end).x,[],3);
pred=squeeze(pr);
prediction(n:sz-1+n,:)=reshape(pred,[sz,512]);
n=n+sz;
    
 end
% seg=reshape(pred,[512 512]);
figure(3);imshow(gather(prediction),[])

%     if opts.prefetch
%       nextBatch = train(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(train))) ;
%       getBatch(imdb, nextBatch) ;
%     end
%     if opts.useGpu
%       im = gpuArray(im) ;
%     end
    
    
% res = vl_simplenn(net, gpuArray(im)) ;




% for i=1:size(res(end).x,2)
%   [score(i),pred(i)] = max(squeeze(res(end).x(1,i,:))) ;
% end
% p=gather(pred);
% predict(j)=p;
% % end
% seg=reshape(pred,[512 512]);
% figure(3);imshow(seg,[])

% function [im, labels] = getBatch2(i, batch)
% % --------------------------------------------------------------------
% im = i.images.data(:,:,batch) ;
% im = 256 * reshape(im, 33, 33, 1, []) ;
% labels = i.images.label(1,batch) ;
