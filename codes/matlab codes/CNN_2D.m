function CNN_2D(imdb,varargin)
setup;
% -------------------------------------------------------------------------
% Initialize CNN
% -------------------------------------------------------------------------

net = initialize2DCNN() ;

% -------------------------------------------------------------------------
% Set Training Parameters
% -------------------------------------------------------------------------

trainOpts.batchSize = 500;
trainOpts.numEpochs = 10 ;
trainOpts.continue = true ;
trainOpts.useGpu = true;
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'data/2DcnnReal' ;
trainOpts = vl_argparse(trainOpts, varargin);

% -------------------------------------------------------------------------
% Take the average Image out
% -------------------------------------------------------------------------
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;
% -------------------------------------------------------------------------
% Convert Input to GPUarray if needed
% -------------------------------------------------------------------------
if trainOpts.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end
% -------------------------------------------------------------------------
% Begin Training using Matconvnet Library
% -------------------------------------------------------------------------
h=@getBatch;
[net,info] = cnn_train2D(net, imdb, h, trainOpts) ;

% -------------------------------------------------------------------------
% Move the CNN back to the CPU if it was trained on the GPU
% -------------------------------------------------------------------------
if trainOpts.useGpu
  net = vl_simplenn_move(net, 'cpu') ;
end

% -------------------------------------------------------------------------
% Move the CNN back to the CPU if it was trained on the GPU
% -------------------------------------------------------------------------
net.layers(end) = [] ;
net.imageMean = imageMean ;
save('data/CNN_2D/phantom_net.mat', '-struct', 'net') ;

% -------------------------------------------------------------------------
% Visualize the learned filters
% -------------------------------------------------------------------------

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.filters),'spacing',2)
axis equal ; title('filters in the first layer') ;

% -------------------------------------------------------------------------
% Function GetBatch
% -------------------------------------------------------------------------

function [im, label] = getBatch(imdb, batch)
% --------------------------------------------------------------------
[a b c]=size(imdb.images.data);
im = 256*imdb.images.data(:,:,batch) ;
im = reshape(im, a, b,1, []) ;
label = imdb.images.labels(1,batch) ;
% label = reshape(label, 1, 1,1, []) ;

%% Applying the trained Model

% -------------------------------------------------------------------------
% Load Test Set
% -------------------------------------------------------------------------
n=1;
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

% -------------------------------------------------------------------------
% Load trained CNN
% -------------------------------------------------------------------------
 net = load('data/CNN_2D/phantom_net.mat') ;
 % -------------------------------------------------------------------------
% Prepare Test Data
% -------------------------------------------------------------------------
[r c h]=size(test);
n=33; % patch dimensions. it is square - must be odd
m=floor(n/2);
pred=zeros(r,c,h);
tic;
for i = 1:h
    B=padarray(test(:,:,i),[m m]);
    C(:,:,i)=im2col(B,[n n],'sliding');
end
    x=reshape(C,[n n 1 (r*c)]);
        % Feed to network
        for j=1:512:262144
            in=x(:,:,:,j:j+511);
            res = vl_simplenn(net, (gpuArray(in))) ;
           
       
    out=gather(res(end).x);

[score,pr] = max(out,[],3);
pred(j,:,i)=squeeze(pr)';
        end
end
    
end
